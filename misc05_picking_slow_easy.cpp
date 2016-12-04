// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <stack>   
#include <sstream>
// Include GLEW
#include <GL/glew.h>
// Include GLFW
#include <glfw3.h>
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
using namespace glm;
// Include AntTweakBar
#include <AntTweakBar.h>

#include <common/shader.hpp>
#include <common/controls.hpp>
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>

#include <iostream>
using namespace std;
#include <fstream>
#include <iterator>

#include "tga.h"
#include "ray_casting.h"

const int window_width = 600, window_height = 600;

typedef struct Vertex {
	float Position[4];
	float Color[4];
	float Normal[3];
	void SetPosition(float *coords) {
		Position[0] = coords[0];
		Position[1] = coords[1];
		Position[2] = coords[2];
		Position[3] = 1.0;
	}
	void SetColor(float *color) {
		Color[0] = color[0];
		Color[1] = color[1];
		Color[2] = color[2];
		Color[3] = color[3];
	}
	void SetNormal(float *coords) {
		Normal[0] = coords[0];
		Normal[1] = coords[1];
		Normal[2] = coords[2];
	}
};

typedef struct point {
    float x, y, z;
    point(const float x = 0, const float y = 0, const float z = 0) : x(x), y(y), z(z) {}
    point(float *coords) : x(coords[0]), y(coords[1]), z(coords[2]) {}
    point operator -(const point& a)const {
        return point(x - a.x, y - a.y, z - a.z);
    }
    point operator +(const point& a)const {
        return point(x + a.x, y + a.y, z + a.z);
    }
    point operator *(const float& a)const {
        return point(x*a, y*a, z*a);
    }
    point operator /(const float& a)const {
        return point(x / a, y / a, z / a);
    }
    float* toArray() {
        float array[] = { x, y, z, 1.0f };
        return array;
    }
};

// function prototypes
int initWindow(void);
void initOpenGL(void);
void loadObject(char*, glm::vec4, Vertex * &, GLushort* &, int);
void createVAOs(Vertex[], GLushort[], int);
void createObjects(void);
void pickObject(void);
void renderScene(void);
void cleanup(void);
static void keyCallback(GLFWwindow*, int, int, int, int);
static void mouseCallback(GLFWwindow*, int, int, int);
void resetScene(void);
void createVAOsForTex(Vertex[], GLushort[], int);
void saveControlPoints(void);
void loadControlPoints(void);
vec3 findClosestPoint(vec3, vec3, vec3, double);
bool rayTest(vec3, vec3, vec3, vec3, double, double);
bool rayTestPoints(Vertex*, vec3, vec3, unsigned int*, double*, double);
void subdivideControlMesh(void);
void move_vertex(void);
void automate_ray(int, int, float, float, float, float);
void rotateRestGrid(void);

// GLOBAL VARIABLES
GLFWwindow* window;

glm::mat4 gProjectionMatrix;
glm::mat4 gViewMatrix;

GLuint gPickedIndex = -1;
std::string gMessage;

GLuint programID;
GLuint pickingProgramID;
GLuint textureProgramID;

const GLuint NumObjects = 256;	// ATTN: THIS NEEDS TO CHANGE AS YOU ADD NEW OBJECTS
GLuint VertexArrayId[NumObjects] = { 0 };
GLuint VertexBufferId[NumObjects] = { 0 };
GLuint IndexBufferId[NumObjects] = { 0 };

size_t NumIndices[NumObjects] = { 0 };
size_t VertexBufferSize[NumObjects] = { 0 };
size_t IndexBufferSize[NumObjects] = { 0 };

GLuint MatrixID;
GLuint ModelMatrixID;
GLuint ViewMatrixID;
GLuint ProjMatrixID;
GLuint PickingMatrixID;
GLuint pickingColorID;
GLuint LightID;
GLuint Light1ID;
GLuint Light2ID;
GLuint LightID_tex;
GLuint Light1ID_tex;
GLuint Light2ID_tex;
GLuint TextureID;

GLint gX = 0.0;
GLint gZ = 0.0;

// animation control
bool animation = false;
GLfloat phi = 0.0;
GLfloat cameraAngleTheta = M_PI/4;
GLfloat cameraAnglePhi = asin(1/sqrt(3));
GLfloat cameraSphereRadius = sqrt(675);
Vertex* Face_Verts;
GLushort* Face_Idcs;
Vertex* hair_Verts;
GLushort* hair_Idcs;
Vertex ControlMeshVerts[441];
GLushort ControlMeshIdcs[1764];
GLushort ControlMeshIdcsForTex[2646];
Vertex ControlMeshSubdivVerts[3721] = {0.0f};
GLushort ControlMeshSubdivIdcs[14884];
GLushort ControlMeshSubdivIdcsForTex[22326];
long image_width;
long image_height;
GLuint texID;
GLfloat uv[882];
GLfloat uvSubdiv[7442];
GLint viewport[4];
vec3 startMousePos;
vec3 endMousePos;
unsigned int id;
double proj;
float colorRed[] = {1.0f, 0.0f, 0.0f, 1.0f};
float controlMeshNormal[] = {0.0f, 0.0f, 1.0f};
glm::mat4 ModelMatrix;

bool moveCameraLeft = false;
bool moveCameraRight = false;
bool moveCameraUp = false;
bool moveCameraDown = false;
bool shouldResetScene = false;
bool shouldDisplayFaceMesh = false;
bool shouldDisplayControlMesh = false;
bool shouldSubdivideControlMesh = false;
bool shouldMoveControlPoint = false;
bool shouldMoveControlPointBackward = false;
bool shouldMoveControlPointForward = false;
bool shouldMoveControlPointLeft = false;
bool shouldMoveControlPointRight = false;
bool shouldMoveControlPointUp = false;
bool shouldMoveControlPointDown = false;
bool shouldLoadHair = false;

void loadObject(char* file, glm::vec4 color, Vertex * &out_Vertices, GLushort* &out_Indices, int ObjectId)
{
	// Read our .obj file
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals;
    bool res = loadOBJ(file, vertices, normals);

	std::vector<GLushort> indices;
	std::vector<glm::vec3> indexed_vertices;
	std::vector<glm::vec2> indexed_uvs;
	std::vector<glm::vec3> indexed_normals;
    indexVBO(vertices, normals, indices, indexed_vertices, indexed_normals);

	const size_t vertCount = indexed_vertices.size();
	const size_t idxCount = indices.size();

	// populate output arrays
	out_Vertices = new Vertex[vertCount];
	for (int i = 0; i < vertCount; i++) {
		out_Vertices[i].SetPosition(&indexed_vertices[i].x);
		out_Vertices[i].SetNormal(&indexed_normals[i].x);
		out_Vertices[i].SetColor(&color[0]);
	}
	out_Indices = new GLushort[idxCount];
	for (int i = 0; i < idxCount; i++) {
		out_Indices[i] = indices[i];
	}

	// set global variables!!
	NumIndices[ObjectId] = idxCount;
	VertexBufferSize[ObjectId] = sizeof(out_Vertices[0]) * vertCount;
	IndexBufferSize[ObjectId] = sizeof(GLushort) * idxCount;
}


void createObjects(void)
{
	//-- COORDINATE AXES --//
	Vertex CoordVerts[] =
	{
		{ { 0.0, 0.0, 0.0, 1.0 }, { 1.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 } },
		{ { 5.0, 0.0, 0.0, 1.0 }, { 1.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 } },
		{ { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 } },
		{ { 0.0, 5.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 } },
		{ { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0, 1.0 }, { 0.0, 0.0, 1.0 } },
		{ { 0.0, 0.0, 5.0, 1.0 }, { 0.0, 0.0, 1.0, 1.0 }, { 0.0, 0.0, 1.0 } },
	};

	VertexBufferSize[0] = sizeof(CoordVerts);	// ATTN: this needs to be done for each hand-made object with the ObjectID (subscript)
	createVAOs(CoordVerts, NULL, 0);
	
	//-- GRID --//
	
	// ATTN: create your grid vertices here!
    Vertex GridVerts[44];
    int k = 0;
    for(int i = -5; i <= 5; i++) {
        GridVerts[4 * k] = {{i, 0, -5, 1.0}, {1.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 1.0}};
        GridVerts[4 * k + 1] = {{i, 0, 5, 1.0}, {1.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 1.0}};
        GridVerts[4 * k + 2] = {{-5, 0, i, 1.0}, {1.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 1.0}};
        GridVerts[4 * k + 3] = {{5, 0, i, 1.0}, {1.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 1.0}};
        k++;
    }

    VertexBufferSize[1] = sizeof(GridVerts);
    createVAOs(GridVerts, NULL, 1);

    k = 0;
    for(int i = -10; i <= 10; i++) {
        for(int j = 0; j <= 20; j++) {
            ControlMeshVerts[21 * k + j] = {{i, j, -5.0, 1.0}, {0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0}};
        }
        cout << k << endl;
        k++;
    }

    texID = load_texture_TGA("biswas_sayak_new.tga", &image_width, &image_height, GL_CLAMP, GL_CLAMP);

    /*glm::vec3 n = glm::vec3(ControlMeshVerts[1].Position[0], ControlMeshVerts[1].Position[1], ControlMeshVerts[1].Position[2]);
    glm::vec3 u = glm::normalize(glm::vec3(ControlMeshVerts[1].Position[1], -ControlMeshVerts[1].Position[0], 0));
    glm::vec3 v = glm::cross(n, u);*/

    for(int i = 0; i < 441; i++) {
        cout << "i " << i << " " << ControlMeshVerts[i].Position[0] << " " << ControlMeshVerts[i].Position[1] << " " << ControlMeshVerts[i].Position[2] << endl;
        /*GLfloat u_coord = glm::dot(u,
                                   glm::vec3(ControlMeshVerts[i].Position[0],
                                    ControlMeshVerts[i].Position[1],
                                    ControlMeshVerts[i].Position[2]));
        GLfloat v_coord = glm::dot(v, glm::vec3(ControlMeshVerts[i].Position[0],
                                   ControlMeshVerts[i].Position[1],
                                   ControlMeshVerts[i].Position[2]));
        cout << u_coord << " " << v_coord << endl;
        uv[2 * i] = u_coord;
        uv[2 * i + 1] = v_coord;*/
        uv[2 * i] = (ControlMeshVerts[i].Position[0] + 10) / 20;
        uv[2 * i + 1] = (ControlMeshVerts[i].Position[1]) / 20;

        if((i + 1) % 21 != 0 && i < 420 && i != 440) {
            ControlMeshIdcs[4 * i] = i;
            ControlMeshIdcs[4 * i + 1] = i + 1;
        } else {
            ControlMeshIdcs[4 * i] = i;
            ControlMeshIdcs[4 * i + 1] = i;
        }
        if(i < 420) {
            ControlMeshIdcs[4 * i + 2] = i;
            ControlMeshIdcs[4 * i + 3] = i + 21;
        } else if(i != 440) {
            ControlMeshIdcs[4 * i + 2] = i;
            ControlMeshIdcs[4 * i + 3] = i + 1;
        }
        if(i == 0 || (i + 1) % 21 != 0 && i < 420) {
            ControlMeshIdcsForTex[6 * i] = i;
            ControlMeshIdcsForTex[6 * i + 1] = i + 1;
            ControlMeshIdcsForTex[6 * i + 2] = i + 22;
            ControlMeshIdcsForTex[6 * i + 3] = i + 22;
            ControlMeshIdcsForTex[6 * i + 4] = i + 21;
            ControlMeshIdcsForTex[6 * i + 5] = i;
        }
    }

    for(int i = 0; i < 881; i+=2) {
        cout << "u " << uv[i] << " v " << uv[i+1] << endl;
    }

    VertexBufferSize[2] = sizeof(ControlMeshVerts);
    IndexBufferSize[2] = sizeof(ControlMeshIdcs);
    createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

    VertexBufferSize[3] = sizeof(ControlMeshVerts);
    IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
    createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
	
	//-- .OBJs --//

	// ATTN: load your models here
	//Vertex* Verts;
	//GLushort* Idcs;
    loadObject("biswas_sayak_scaled.obj", glm::vec4(1.0, 1.0, 1.0, 1.0), Face_Verts, Face_Idcs, 4);
    createVAOs(Face_Verts, Face_Idcs, 4);

    loadObject("hair.obj", glm::vec4(0.3, 0.5, 0.2, 1.0), hair_Verts, hair_Idcs, 7);
    createVAOs(hair_Verts, hair_Idcs, 7);
}

void renderScene(void)
{
	//ATTN: DRAW YOUR SCENE HERE. MODIFY/ADAPT WHERE NECESSARY!


	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.2f, 0.0f);
	// Re-clear the screen for real rendering
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if(moveCameraLeft && !shouldMoveControlPoint) {
        cameraAngleTheta -= 0.01f;
    }

    if(moveCameraRight && !shouldMoveControlPoint) {
        cameraAngleTheta += 0.01f;
    }

    if(moveCameraUp && !shouldMoveControlPoint) {
        cameraAnglePhi -= 0.01f;
    }

    if(moveCameraDown && !shouldMoveControlPoint) {
        cameraAnglePhi += 0.01f;
    }

    if(moveCameraLeft || moveCameraRight || moveCameraDown || moveCameraUp || shouldResetScene) {
        float camX = cameraSphereRadius * cos(cameraAnglePhi) * sin(cameraAngleTheta);
        float camY = cameraSphereRadius * sin(cameraAnglePhi);
        float camZ = cameraSphereRadius * cos(cameraAnglePhi) * cos(cameraAngleTheta);
        gViewMatrix = glm::lookAt(glm::vec3(camX, camY, camZ),	// eye
            glm::vec3(0.0, 10.0, 0.0),	// center
            glm::vec3(0.0, 1.0, 0.0));	// up
    }

    if(shouldMoveControlPointBackward) {
        float coords[] = {ControlMeshVerts[id].Position[0], ControlMeshVerts[id].Position[1], ControlMeshVerts[id].Position[2] + 0.01f};
        ControlMeshVerts[id].SetPosition(coords);
        VertexBufferSize[2] = sizeof(ControlMeshVerts);
        IndexBufferSize[2] = sizeof(ControlMeshIdcs);
        createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

        VertexBufferSize[3] = sizeof(ControlMeshVerts);
        IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
        createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
    }

    if(shouldMoveControlPointForward) {
        float coords[] = {ControlMeshVerts[id].Position[0], ControlMeshVerts[id].Position[1], ControlMeshVerts[id].Position[2] - 0.01f};
        ControlMeshVerts[id].SetPosition(coords);
        VertexBufferSize[2] = sizeof(ControlMeshVerts);
        IndexBufferSize[2] = sizeof(ControlMeshIdcs);
        createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

        VertexBufferSize[3] = sizeof(ControlMeshVerts);
        IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
        createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
    }

    if(shouldMoveControlPointLeft) {
        float coords[] = {ControlMeshVerts[id].Position[0] - 0.01f, ControlMeshVerts[id].Position[1], ControlMeshVerts[id].Position[2]};
        ControlMeshVerts[id].SetPosition(coords);
        VertexBufferSize[2] = sizeof(ControlMeshVerts);
        IndexBufferSize[2] = sizeof(ControlMeshIdcs);
        createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

        VertexBufferSize[3] = sizeof(ControlMeshVerts);
        IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
        createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
    }

    if(shouldMoveControlPointRight) {
        float coords[] = {ControlMeshVerts[id].Position[0] + 0.01f, ControlMeshVerts[id].Position[1], ControlMeshVerts[id].Position[2]};
        ControlMeshVerts[id].SetPosition(coords);
        VertexBufferSize[2] = sizeof(ControlMeshVerts);
        IndexBufferSize[2] = sizeof(ControlMeshIdcs);
        createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

        VertexBufferSize[3] = sizeof(ControlMeshVerts);
        IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
        createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
    }

    if(shouldMoveControlPointUp) {
        float coords[] = {ControlMeshVerts[id].Position[0], ControlMeshVerts[id].Position[1] + 0.01f, ControlMeshVerts[id].Position[2]};
        ControlMeshVerts[id].SetPosition(coords);
        VertexBufferSize[2] = sizeof(ControlMeshVerts);
        IndexBufferSize[2] = sizeof(ControlMeshIdcs);
        createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

        VertexBufferSize[3] = sizeof(ControlMeshVerts);
        IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
        createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
    }

    if(shouldMoveControlPointDown) {
        float coords[] = {ControlMeshVerts[id].Position[0], ControlMeshVerts[id].Position[1] - 0.01f, ControlMeshVerts[id].Position[2]};
        ControlMeshVerts[id].SetPosition(coords);
        VertexBufferSize[2] = sizeof(ControlMeshVerts);
        IndexBufferSize[2] = sizeof(ControlMeshIdcs);
        createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

        VertexBufferSize[3] = sizeof(ControlMeshVerts);
        IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
        createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
    }

	glUseProgram(programID);
	{
        glm::vec3 lightPos = glm::vec3(0, 20, -10);
        glm::vec3 light1Pos = glm::vec3(10, 20, -10);
        glm::vec3 light2Pos = glm::vec3(-10, 20, -10);
		glm::mat4x4 ModelMatrix = glm::mat4(1.0);
		glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);
        glUniform3f(Light1ID, light1Pos.x, light1Pos.y, light1Pos.z);
        glUniform3f(Light2ID, light2Pos.x, light2Pos.y, light2Pos.z);
		glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &gViewMatrix[0][0]);
		glUniformMatrix4fv(ProjMatrixID, 1, GL_FALSE, &gProjectionMatrix[0][0]);
		glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);

		glBindVertexArray(VertexArrayId[0]);	// draw CoordAxes
		glDrawArrays(GL_LINES, 0, 6);

        glBindVertexArray(VertexArrayId[1]);
        glDrawArrays(GL_LINES, 0, 44);

        if(shouldDisplayControlMesh) {
            glDisable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_POINT_SIZE);
            glPointSize(7.0f);
            glBindVertexArray(VertexArrayId[2]);
            glDrawArrays(GL_POINTS, 0, 441);
            //glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
            glDrawElements(GL_LINES, 1764, GL_UNSIGNED_SHORT, (GLvoid*)0);
        }

        if(shouldDisplayFaceMesh) {
            glBindVertexArray(VertexArrayId[4]);
            glDrawElements(GL_TRIANGLES, NumIndices[4], GL_UNSIGNED_SHORT, (void*)0);
        }

        if(shouldSubdivideControlMesh) {
            glDisable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_POINT_SIZE);
            glPointSize(4.0f);
            glBindVertexArray(VertexArrayId[5]);
            glDrawArrays(GL_POINTS, 0, 3721);
            glDrawElements(GL_LINES, 14884, GL_UNSIGNED_SHORT, (GLvoid*)0);
        }

        if(shouldLoadHair) {
            glBindVertexArray(VertexArrayId[7]);
            glDrawElements(GL_TRIANGLES, NumIndices[7], GL_UNSIGNED_SHORT, (void*)0);
        }
			
		glBindVertexArray(0);
	}

    if(shouldDisplayControlMesh || shouldSubdivideControlMesh) {
        glUseProgram(textureProgramID); {
            glm::vec3 lightPos = glm::vec3(0, 20, -10);
            glm::vec3 light1Pos = glm::vec3(10, 20, -10);
            glm::vec3 light2Pos = glm::vec3(-10, 20, -10);
            glm::mat4x4 ModelMatrix = glm::mat4(1.0);
            glUniform3f(LightID_tex, lightPos.x, lightPos.y, lightPos.z);
            glUniform3f(Light1ID_tex, light1Pos.x, light1Pos.y, light1Pos.z);
            glUniform3f(Light2ID_tex, light2Pos.x, light2Pos.y, light2Pos.z);
            glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &gViewMatrix[0][0]);
            glUniformMatrix4fv(ProjMatrixID, 1, GL_FALSE, &gProjectionMatrix[0][0]);
            glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texID);
            glUniform1i(TextureID, 0);

            /*glDisable(GL_PROGRAM_POINT_SIZE);
            glEnable(GL_POINT_SIZE);
            glPointSize(7.0f);*/
            glBindVertexArray(VertexArrayId[3]);
            glDrawElements(GL_TRIANGLES, 2646, GL_UNSIGNED_SHORT, (GLvoid*)0);

            glBindVertexArray(0);
        }
    }

    glUseProgram(0);
	// Draw GUI
    //TwDraw();

	// Swap buffers
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void pickObject(void)
{
	// Clear the screen in white
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(pickingProgramID);
	{
        ModelMatrix = glm::mat4(1.0); // TranslationMatrix * RotationMatrix;
		glm::mat4 MVP = gProjectionMatrix * gViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader, in the "MVP" uniform
		glUniformMatrix4fv(PickingMatrixID, 1, GL_FALSE, &MVP[0][0]);
		
		// ATTN: DRAW YOUR PICKING SCENE HERE. REMEMBER TO SEND IN A DIFFERENT PICKING COLOR FOR EACH OBJECT BEFOREHAND
		glBindVertexArray(0);

	}
	glUseProgram(0);
	// Wait until all the pending drawing commands are really done.
	// Ultra-mega-over slow ! 
	// There are usually a long time between glDrawElements() and
	// all the fragments completely rasterized.
	glFlush();
	glFinish();

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Read the pixel at the center of the screen.
	// You can also use glfwGetMousePos().
	// Ultra-mega-over slow too, even for 1 pixel, 
	// because the framebuffer is on the GPU.
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    mat4 nModelMatrix = gViewMatrix * ModelMatrix;

	unsigned char data[4];
    GLfloat zpos;

	glReadPixels(xpos, window_height - ypos, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, data); // OpenGL renders with (0,0) on bottom, mouse reports with (0,0) on top

    startMousePos = glm::unProject(glm::vec3(xpos, window_height - ypos, 0.0f), nModelMatrix, gProjectionMatrix, vec4(viewport[0], viewport[1], viewport[2], viewport[3]));
    endMousePos = glm::unProject(glm::vec3(xpos, window_height - ypos, 1.0f), nModelMatrix, gProjectionMatrix, vec4(viewport[0], viewport[1], viewport[2], viewport[3]));
    double epsilon = 0.1;
    double proj;
    bool found = rayTestPoints(ControlMeshVerts, startMousePos, endMousePos, &id, &proj, epsilon);
    if(id > 440) {
        id = 0;
    }
    cout << "id: " << id << endl;

	// Convert the color back to an integer ID
	gPickedIndex = int(data[0]);
	
	if (gPickedIndex == 255){ // Full white, must be the background !
		gMessage = "background";
	}
	else {
		std::ostringstream oss;
		oss << "point " << gPickedIndex;
		gMessage = oss.str();
	}

	// Uncomment these lines to see the picking shader in effect
	//glfwSwapBuffers(window);
	//continue; // skips the normal rendering
}

void move_vertex() {

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glm::vec4 vp = glm::vec4(viewport[0], viewport[1], viewport[2], viewport[3]);

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    mat4 nModelMatrix = gViewMatrix * ModelMatrix;


    int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    vec3 W;
    if (state == GLFW_PRESS) {
        cout << "When pressed" << W.z << endl;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            vec3 p = glm::project(glm::vec3(ControlMeshVerts[id].Position[0], ControlMeshVerts[id].Position[1], ControlMeshVerts[id].Position[2]), nModelMatrix, gProjectionMatrix, glm::vec4(viewport[0], viewport[1], viewport[2], viewport[3]));
            W = glm::unProject(glm::vec3(xpos, window_height - ypos, p.z), nModelMatrix, gProjectionMatrix, glm::vec4(viewport[0], viewport[1], viewport[2], viewport[3]));
            float coords[3] = { W.x, W.y, W.z };
            ControlMeshVerts[id].SetPosition(coords);
        }
        else {
            vec3 p = glm::project(glm::vec3(ControlMeshVerts[id].Position[0], ControlMeshVerts[id].Position[1], ControlMeshVerts[id].Position[2]), nModelMatrix, gProjectionMatrix, glm::vec4(viewport[0], viewport[1], viewport[2], viewport[3]));
            W = glm::unProject(glm::vec3(xpos, window_height - ypos, p.z), nModelMatrix, gProjectionMatrix, glm::vec4(viewport[0], viewport[1], viewport[2], viewport[3]));
            float coords[3] = { W.x, W.y, ControlMeshVerts[id].Position[2] };
            ControlMeshVerts[id].SetPosition(coords);
        }
    }

    VertexBufferSize[2] = sizeof(ControlMeshVerts);
    IndexBufferSize[2] = sizeof(ControlMeshIdcs);
    createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

    VertexBufferSize[3] = sizeof(ControlMeshVerts);
    IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
    createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
}

vec3 findClosestPoint(vec3 rayStartPos, vec3 rayEndPos, vec3 pointPos, double *proj) {
    vec3 rayVector = rayEndPos - rayStartPos;
    double raySquared = glm::dot(rayVector, rayVector);
    vec3 projection = pointPos - rayStartPos;
    double projectionVal = glm::dot(projection, rayVector);
    *proj = projectionVal / raySquared;
    vec3 closestPoint = rayStartPos + glm::vec3(rayVector.x * (*proj), rayVector.y * (*proj), rayVector.z * (*proj)) ;
    return closestPoint;
}

bool rayTest(vec3 pointPos, vec3 startPos, vec3 endPos, vec3 *closestPoint, double *proj, double epsilon) {
    *closestPoint = findClosestPoint(startPos, endPos, pointPos, proj);
    //cout << "closestPoint " << closestPoint->x << " " << closestPoint->y << " " << closestPoint->z << endl;
    double len = glm::distance2(*closestPoint, pointPos);
    //cout << "len " << len << endl;
    return len < epsilon;
}

bool rayTestPoints(Vertex* vert, vec3 start, vec3 end, unsigned int *id, double *proj, double epsilon) {
    unsigned int pointID = 442;
    bool foundCollision = false;
    double minDistToStart = 10000000.0;
    double distance;
    vec3 point;
    for (unsigned int i = 0; i < 441; ++i) {
        //cout << "i " << i << endl;
        vec3 pointPos = glm::vec3(vert[i].Position[0], vert[i].Position[1], vert[i].Position[2]);
        if (rayTest(pointPos, start, end, &point, proj, epsilon)) {
            distance = glm::distance2(start, point);
            //cout << "distance " << distance << endl;
            //cout << "pointPos " << pointPos.x << " " << pointPos.y << " " << pointPos.z << endl;
            if (distance < minDistToStart)
            {
                minDistToStart = distance;
                pointID = i;
                foundCollision = true;
            }
        }
    }

    *id = pointID;
    return foundCollision;
}

void automate_ray(int range_min, int range_max, float dir_x, float dir_y, float dir_z, float face_verts_cutoff) {
    float cg[3];
    int coord_index = 0;
    if(dir_x == 1) {
        coord_index = 0;
    } else if(dir_y == 1) {
        coord_index = 1;
    } else if(dir_z == 1) {
        coord_index = 2;
    }

    bool condition = false;

    for(int j = range_min; j < range_max; j++)
    {
        for (int i = NumIndices[4] - 1; i > 0; i = i - 3) {
            if(dir_z != 0) {
                condition = Face_Verts[Face_Idcs[i]].Position[0] < ControlMeshVerts[j].Position[0] + 5.0f
                        && Face_Verts[Face_Idcs[i]].Position[0] > ControlMeshVerts[j].Position[0] - 5.0f
                        && Face_Verts[Face_Idcs[i]].Position[1] < ControlMeshVerts[j].Position[1] + 5.0f
                        && Face_Verts[Face_Idcs[i]].Position[1] > ControlMeshVerts[j].Position[1] - 5.0f
                        && Face_Verts[Face_Idcs[i]].Position[coord_index] < face_verts_cutoff;
            } else if(dir_x != 0) {
                if(face_verts_cutoff > 0) {
                    condition = Face_Verts[Face_Idcs[i]].Position[0] < ControlMeshVerts[j].Position[0] + 5.0f
                            && Face_Verts[Face_Idcs[i]].Position[0] > ControlMeshVerts[j].Position[0] - 5.0f
                            && Face_Verts[Face_Idcs[i]].Position[1] < ControlMeshVerts[j].Position[1] + 5.0f
                            && Face_Verts[Face_Idcs[i]].Position[1] > ControlMeshVerts[j].Position[1] - 5.0f
                            && Face_Verts[Face_Idcs[i]].Position[coord_index] > face_verts_cutoff;
                } else {
                    condition = Face_Verts[Face_Idcs[i]].Position[0] < ControlMeshVerts[j].Position[0] + 5.0f
                            && Face_Verts[Face_Idcs[i]].Position[0] > ControlMeshVerts[j].Position[0] - 5.0f
                            && Face_Verts[Face_Idcs[i]].Position[1] < ControlMeshVerts[j].Position[1] + 5.0f
                            && Face_Verts[Face_Idcs[i]].Position[1] > ControlMeshVerts[j].Position[1] - 5.0f
                            && Face_Verts[Face_Idcs[i]].Position[coord_index] < face_verts_cutoff;
                }
            } else if(dir_y != 0) {
                condition = Face_Verts[Face_Idcs[i]].Position[0] < ControlMeshVerts[j].Position[0] + 5.0f
                        && Face_Verts[Face_Idcs[i]].Position[0] > ControlMeshVerts[j].Position[0] - 5.0f
                        && Face_Verts[Face_Idcs[i]].Position[1] < ControlMeshVerts[j].Position[1] + 5.0f
                        && Face_Verts[Face_Idcs[i]].Position[1] > ControlMeshVerts[j].Position[1] - 5.0f
                        && Face_Verts[Face_Idcs[i]].Position[coord_index] > face_verts_cutoff;
            }

            if (condition) {
                const float v1[3] = { Face_Verts[Face_Idcs[i]].Position[0] ,Face_Verts[Face_Idcs[i]].Position[1],Face_Verts[Face_Idcs[i]].Position[2] };
                const float v2[3] = { Face_Verts[Face_Idcs[i + 1]].Position[0] ,Face_Verts[Face_Idcs[i + 1]].Position[1],Face_Verts[Face_Idcs[i + 1]].Position[2] };
                const float v3[3] = { Face_Verts[Face_Idcs[i + 2]].Position[0] ,Face_Verts[Face_Idcs[i + 2]].Position[1],Face_Verts[Face_Idcs[i + 2]].Position[2] };
                const float p[3] = { ControlMeshVerts[j].Position[0] ,ControlMeshVerts[j].Position[1],ControlMeshVerts[j].Position[2] };
                const float dir[3] = { dir_x, dir_y , dir_z };
                ray_cast(v1, v2, v3, p, dir, cg);
                if (cg[0] >= 0 && cg[1] >= 0 && cg[2] >= 0)
                {
                    if(dir_x != 0) {
                        float newX = (Face_Verts[Face_Idcs[i]].Position[0] + Face_Verts[Face_Idcs[i + 1]].Position[0] + Face_Verts[Face_Idcs[i + 2]].Position[0]) / 3.0f;
                        float newval[3] = { newX - 0.5f, ControlMeshVerts[j].Position[1], ControlMeshVerts[j].Position[2]};
                        ControlMeshVerts[j].SetPosition(newval);
                    }

                    if(dir_y != 0) {
                        float newY = (Face_Verts[Face_Idcs[i]].Position[1] + Face_Verts[Face_Idcs[i + 1]].Position[1] + Face_Verts[Face_Idcs[i + 2]].Position[1]) / 3.0f;
                        float newval[3] = { ControlMeshVerts[j].Position[0], newY - 0.5f, ControlMeshVerts[j].Position[2]};
                        ControlMeshVerts[j].SetPosition(newval);
                    }

                    if(dir_z != 0) {
                        float newZ = (Face_Verts[Face_Idcs[i]].Position[2] + Face_Verts[Face_Idcs[i + 1]].Position[2] + Face_Verts[Face_Idcs[i + 2]].Position[2]) / 3.0f;
                        float newval[3] = { ControlMeshVerts[j].Position[0], ControlMeshVerts[j].Position[1], newZ - 0.5f};
                        ControlMeshVerts[j].SetPosition(newval);
                    }

                    break;
                }
            } else {
                if(dir_z != 0) {
                    float newval[3] = { ControlMeshVerts[j].Position[0], ControlMeshVerts[j].Position[1], 2.0f };
                    ControlMeshVerts[j].SetPosition(newval);
                }

                /*if(dir_x != 0) {
                    float newval[3] = { face_verts_cutoff, ControlMeshVerts[j].Position[1], ControlMeshVerts[j].Position[2] };
                    ControlMeshVerts[j].SetPosition(newval);
                }*/

                if(dir_y != 0) {
                    float newval[3] = { ControlMeshVerts[j].Position[0], face_verts_cutoff, ControlMeshVerts[j].Position[2] };
                    ControlMeshVerts[j].SetPosition(newval);
                }
            }
        }
        VertexBufferSize[2] = sizeof(ControlMeshVerts);
        IndexBufferSize[2] = sizeof(ControlMeshIdcs);
        createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

        VertexBufferSize[3] = sizeof(ControlMeshVerts);
        IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
        createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
    }

    //cout << "barycentric:" << cg[0] << " " << cg[1] << " " << cg[2] << endl;
}

int initWindow(void)
{
	// Initialise GLFW
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	// Open a window and create its OpenGL context
    window = glfwCreateWindow(window_width, window_height, "Biswas,Sayak(54584911)", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	// Initialize the GUI
    TwInit(TW_OPENGL_CORE, NULL);
	TwWindowSize(window_width, window_height);
	TwBar * GUI = TwNewBar("Picking");
	TwSetParam(GUI, NULL, "refresh", TW_PARAM_CSTRING, 1, "0.1");
    TwAddVarRW(GUI, "Last picked object", TW_TYPE_STDSTRING, &gMessage, NULL);

	// Set up inputs
	glfwSetCursorPos(window, window_width / 2, window_height / 2);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetMouseButtonCallback(window, mouseCallback);

	return 0;
}

void initOpenGL(void)
{

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    gProjectionMatrix = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 100.0f);
	// Or, for an ortho camera :
    //gProjectionMatrix = glm::ortho(-4.0f, 4.0f, -3.0f, 3.0f, 0.0f, 100.0f); // In world coordinates

	// Camera matrix
    gViewMatrix = glm::lookAt(glm::vec3(15.0, 15.0, 15.0f),	// eye
        glm::vec3(0.0, 10.0, 0.0),	// center
		glm::vec3(0.0, 1.0, 0.0));	// up

	// Create and compile our GLSL program from the shaders
	programID = LoadShaders("StandardShading.vertexshader", "StandardShading.fragmentshader");
	pickingProgramID = LoadShaders("Picking.vertexshader", "Picking.fragmentshader");
    textureProgramID = LoadShaders("TextureShading.vertexshader", "TextureShading.fragmentshader");

	// Get a handle for our "MVP" uniform
	MatrixID = glGetUniformLocation(programID, "MVP");
	ModelMatrixID = glGetUniformLocation(programID, "M");
	ViewMatrixID = glGetUniformLocation(programID, "V");
	ProjMatrixID = glGetUniformLocation(programID, "P");
	
	PickingMatrixID = glGetUniformLocation(pickingProgramID, "MVP");
	// Get a handle for our "pickingColorID" uniform
	pickingColorID = glGetUniformLocation(pickingProgramID, "PickingColor");
	// Get a handle for our "LightPosition" uniform
	LightID = glGetUniformLocation(programID, "LightPosition_worldspace");
    Light1ID = glGetUniformLocation(programID, "Light1Position_worldspace");
    Light2ID = glGetUniformLocation(programID, "Light2Position_worldspace");

    LightID_tex = glGetUniformLocation(textureProgramID, "LightPosition_worldspace");
    Light1ID_tex = glGetUniformLocation(textureProgramID, "Light1Position_worldspace");
    Light2ID_tex = glGetUniformLocation(textureProgramID, "Light2Position_worldspace");

    TextureID = glGetUniformLocation(textureProgramID, "myTextureSampler");

	createObjects();
}

void createVAOs(Vertex Vertices[], unsigned short Indices[], int ObjectId) {

	GLenum ErrorCheckValue = glGetError();
	const size_t VertexSize = sizeof(Vertices[0]);
	const size_t RgbOffset = sizeof(Vertices[0].Position);
	const size_t Normaloffset = sizeof(Vertices[0].Color) + RgbOffset;

	// Create Vertex Array Object
	glGenVertexArrays(1, &VertexArrayId[ObjectId]);	//
	glBindVertexArray(VertexArrayId[ObjectId]);		//

	// Create Buffer for vertex data
	glGenBuffers(1, &VertexBufferId[ObjectId]);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[ObjectId]);
	glBufferData(GL_ARRAY_BUFFER, VertexBufferSize[ObjectId], Vertices, GL_STATIC_DRAW);

	// Create Buffer for indices
	if (Indices != NULL) {
		glGenBuffers(1, &IndexBufferId[ObjectId]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[ObjectId]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexBufferSize[ObjectId], Indices, GL_STATIC_DRAW);
	}

	// Assign vertex attributes
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset); 
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)Normaloffset);

	glEnableVertexAttribArray(0);	// position
	glEnableVertexAttribArray(1);	// color
	glEnableVertexAttribArray(2);	// normal

	// Disable our Vertex Buffer Object 
	glBindVertexArray(0);

	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR)
	{
		fprintf(
			stderr,
			"ERROR: Could not create a VBO: %s \n",
			gluErrorString(ErrorCheckValue)
			);
	}
}

void createVAOsForTex(Vertex Vertices[], unsigned short Indices[], int ObjectId) {
    GLenum ErrorCheckValue = glGetError();
    const size_t VertexSize = sizeof(Vertices[0]);
    const size_t RgbOffset = sizeof(Vertices[0].Position);
    const size_t Normaloffset = sizeof(Vertices[0].Color) + RgbOffset;

    // Create Vertex Array Object
    glGenVertexArrays(1, &VertexArrayId[ObjectId]);	//
    glBindVertexArray(VertexArrayId[ObjectId]);		//

    // Create Buffer for vertex data
    glGenBuffers(1, &VertexBufferId[ObjectId]);
    glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[ObjectId]);
    glBufferData(GL_ARRAY_BUFFER, VertexBufferSize[ObjectId], Vertices, GL_STATIC_DRAW);

    // Create Buffer for indices
    if (Indices != NULL) {
        glGenBuffers(1, &IndexBufferId[ObjectId]);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[ObjectId]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexBufferSize[ObjectId], Indices, GL_STATIC_DRAW);
    }

    // Assign vertex attributes
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, VertexSize, 0);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)Normaloffset);

    glEnableVertexAttribArray(0);	// position
    //glEnableVertexAttribArray(2);	// normal

    if(ObjectId == 6) {
        cout << "object id is 6" << endl;
        GLuint uvbuffer;
        glGenBuffers(1, &uvbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uvSubdiv), uvSubdiv, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
        glEnableVertexAttribArray(1);	// color
    } else {
        cout << "object id is: " << ObjectId << endl;
        GLuint uvbuffer;
        glGenBuffers(1, &uvbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uv), uv, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
        glEnableVertexAttribArray(1);	// color
    }

    // Disable our Vertex Buffer Object
    glBindVertexArray(0);

    ErrorCheckValue = glGetError();
    if (ErrorCheckValue != GL_NO_ERROR)
    {
        fprintf(
            stderr,
            "ERROR: Could not create a VBO: %s \n",
            gluErrorString(ErrorCheckValue)
            );
    }
}

void cleanup(void)
{
	// Cleanup VBO and shader
	for (int i = 0; i < NumObjects; i++) {
		glDeleteBuffers(1, &VertexBufferId[i]);
		glDeleteBuffers(1, &IndexBufferId[i]);
		glDeleteVertexArrays(1, &VertexArrayId[i]);
	}
	glDeleteProgram(programID);
	glDeleteProgram(pickingProgramID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// ATTN: MODIFY AS APPROPRIATE
	if (action == GLFW_PRESS) {
		switch (key)
		{
		case GLFW_KEY_A:
			break;
		case GLFW_KEY_D:
			break;
		case GLFW_KEY_W:
			break;
        case GLFW_KEY_M:
            if(shouldMoveControlPoint) {
                shouldMoveControlPoint = false;
            } else {
                shouldMoveControlPoint = true;
            }
            break;
        case GLFW_KEY_R:
            shouldResetScene = true;
            resetScene();
            break;
		case GLFW_KEY_S:
			break;
		case GLFW_KEY_SPACE:
			break;
        case GLFW_KEY_LEFT:
            moveCameraLeft = true;
            break;
        case GLFW_KEY_RIGHT:
            moveCameraRight = true;
            break;
        case GLFW_KEY_UP:
            moveCameraUp = true;
            break;
        case GLFW_KEY_DOWN:
            moveCameraDown = true;
            break;
        case GLFW_KEY_COMMA:
            shouldMoveControlPointBackward = true;
            break;
        case GLFW_KEY_PERIOD:
            shouldMoveControlPointForward = true;
            break;
        case GLFW_KEY_9:
            shouldMoveControlPointLeft = true;
            break;
        case GLFW_KEY_0:
            shouldMoveControlPointRight = true;
            break;
        case GLFW_KEY_LEFT_BRACKET:
            shouldMoveControlPointUp = true;
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            shouldMoveControlPointDown = true;
            break;
		default:
			break;
		}
    } else if(action == GLFW_RELEASE) {
        switch (key) {
        case GLFW_KEY_A:
            shouldSubdivideControlMesh = true;
            subdivideControlMesh();
            break;
        case GLFW_KEY_C:
            if(shouldDisplayControlMesh) {
                shouldDisplayControlMesh = false;
            } else {
                shouldDisplayControlMesh = true;
            }
            break;
        case GLFW_KEY_F:
            if(shouldDisplayFaceMesh) {
                shouldDisplayFaceMesh = false;
            } else {
                shouldDisplayFaceMesh = true;
            }
            break;
        case GLFW_KEY_H:
            if(shouldLoadHair) {
                shouldLoadHair = false;
            } else {
                shouldLoadHair = true;
            }
            break;
        case GLFW_KEY_L:
            loadControlPoints();
            break;
        case GLFW_KEY_M:
            if(shouldMoveControlPoint) {
                shouldMoveControlPoint = false;
            } else {
                shouldMoveControlPoint = true;
            }
            break;
        case GLFW_KEY_R:
            shouldResetScene = false;
            break;
        case GLFW_KEY_S:
            saveControlPoints();
            break;
        case GLFW_KEY_T:
            rotateRestGrid();
            break;
        case GLFW_KEY_Z:
            automate_ray(0, 441, 0, 0, 1, 3.0f);
            break;
        case GLFW_KEY_LEFT:
            moveCameraLeft = false;
            break;
        case GLFW_KEY_RIGHT:
            moveCameraRight = false;
            break;
        case GLFW_KEY_UP:
            moveCameraUp = false;
            break;
        case GLFW_KEY_DOWN:
            moveCameraDown = false;
            break;
        case GLFW_KEY_COMMA:
            shouldMoveControlPointBackward = false;
            break;
        case GLFW_KEY_PERIOD:
            shouldMoveControlPointForward = false;
            break;
        case GLFW_KEY_9:
            shouldMoveControlPointLeft = false;
            break;
        case GLFW_KEY_0:
            shouldMoveControlPointRight = false;
            break;
        case GLFW_KEY_LEFT_BRACKET:
            shouldMoveControlPointUp = false;
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            shouldMoveControlPointDown = false;
            break;
        default:
            break;
        }
    }
}

static void mouseCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		pickObject();
	}
}

void resetScene(void) {
    cameraAngleTheta = M_PI/4;
    cameraAnglePhi = asin(1/sqrt(3));
}

void saveControlPoints() {
    cout << "Writing Control Point file ..." << endl;
    ofstream controlPointFile;
    controlPointFile.open("cm.p3", ios::out);
    if(controlPointFile.is_open()) {
        for(int i = 0; i < 441; i++) {
            controlPointFile << "v ";
            controlPointFile << to_string(ControlMeshVerts[i].Position[0]) + " ";
            controlPointFile << to_string(ControlMeshVerts[i].Position[1]) + " ";
            controlPointFile << to_string(ControlMeshVerts[i].Position[2]) + " ";
            controlPointFile << to_string(ControlMeshVerts[i].Position[3]) << endl;
        }

        for(int i = 0; i < 441; i++) {
            controlPointFile << "f ";
            controlPointFile << to_string(ControlMeshIdcsForTex[6 * i]) + " ";
            controlPointFile << to_string(ControlMeshIdcsForTex[6 * i + 1]) + " ";
            controlPointFile << to_string(ControlMeshIdcsForTex[6 * i + 2]) + " ";
            controlPointFile << to_string(ControlMeshIdcsForTex[6 * i + 3]) + " ";
            controlPointFile << to_string(ControlMeshIdcsForTex[6 * i + 4]) + " ";
            controlPointFile << to_string(ControlMeshIdcsForTex[6 * i + 5]) << endl;
        }
        cout << "Control points written to cm.p3" << endl;
    } else {
        cout << "Unable to open control point file." << endl;
    }
    controlPointFile.close();
}

void loadControlPoints() {
    cout << "Loading Control Point file ..." << endl;
    ifstream controlPointFile;
    controlPointFile.open("cm.p3", ios::in);
    if(controlPointFile.is_open()) {
        int vert_idx = 0;
        int idcs_idx = 0;
        while(controlPointFile) {
            string line;
            getline(controlPointFile, line);
            if(!line.empty()) {
                istringstream input_string_stream(line);
                vector<string> tokens {istream_iterator<string>{input_string_stream},
                                      istream_iterator<string>{}};
                if(tokens.at(0).compare("v") == 0) {
                    ControlMeshVerts[vert_idx].Position[0] = atof(tokens.at(1).c_str());
                    ControlMeshVerts[vert_idx].Position[1] = atof(tokens.at(2).c_str());
                    ControlMeshVerts[vert_idx].Position[2] = atof(tokens.at(3).c_str());
                    ControlMeshVerts[vert_idx].Position[3] = atof(tokens.at(4).c_str());
                    vert_idx++;
                }

                if(tokens.at(0).compare("f") == 0) {
                    ControlMeshIdcsForTex[6 * idcs_idx] = atoi(tokens.at(1).c_str());
                    ControlMeshIdcsForTex[6 * idcs_idx + 1] = atoi(tokens.at(2).c_str());
                    ControlMeshIdcsForTex[6 * idcs_idx + 2] = atoi(tokens.at(3).c_str());
                    ControlMeshIdcsForTex[6 * idcs_idx + 3] = atoi(tokens.at(4).c_str());
                    ControlMeshIdcsForTex[6 * idcs_idx + 4] = atoi(tokens.at(5).c_str());
                    ControlMeshIdcsForTex[6 * idcs_idx + 5] = atoi(tokens.at(6).c_str());
                    idcs_idx++;
                }
            }
        }
        cout << "Control Points loaded from cm.p3" << endl;
    } else {
        cout << "Unable to open control point file." << endl;
    }
    controlPointFile.close();
    VertexBufferSize[2] = sizeof(ControlMeshVerts);
    IndexBufferSize[2] = sizeof(ControlMeshIdcs);
    createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

    VertexBufferSize[3] = sizeof(ControlMeshVerts);
    IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
    createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
}

void subdivideControlMesh() {
    int j = 0;
    for(int i = 0; i < 441; i++) {
        //cout << "i " << i <<" j " << j << endl;
        point *s00, *s01, *s02, *s10, *s11, *s12, *s20, *s21, *s22;
        float u00, v00, u01, v01, u02, v02, u10, v10, u11, v11, u12, v12, u20, v20, u21, v21, u22, v22;
        if(i < 21) {
            //cout << "in left only" << endl;
            s00 = new point(ControlMeshVerts[i].Position[0] - 1, ControlMeshVerts[i].Position[1] - 1, ControlMeshVerts[i].Position[2]);
            u00 = uv[2 * i] - 1;
            v00 = uv[2 * i + 1] - 1;
            s01 = new point(ControlMeshVerts[i].Position[0] - 1, ControlMeshVerts[i].Position[1], ControlMeshVerts[i].Position[2]);
            u01 = uv[2 * i] - 1;
            v01 = uv[2 * i + 1];
            s02 = new point(ControlMeshVerts[i].Position[0] - 1, ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
            u02 = uv[2 * i] - 1;
            v02 = uv[2 * i + 1] + 1;
            if(i == 0) {
                s10 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1] - 1, ControlMeshVerts[i].Position[2]);
                u10 = uv[2 * i];
                v10 = uv[2 * i + 1] - 1;
            } else {
                s10 = new point(ControlMeshVerts[i - 1].Position[0], ControlMeshVerts[i - 1].Position[1], ControlMeshVerts[i - 1].Position[2]);
                u10 = uv[2 * (i - 1)];
                v10 = uv[2 * (i - 1) + 1];
            }
            s11 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1], ControlMeshVerts[i].Position[2]);
            u11 = uv[2 * i];
            v11 = uv[2 * i + 1];
            if(i == 20) {
                s12 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
                u12 = uv[2 * i];
                v12 = uv[2 * i + 1] + 1;
            } else {
                s12 = new point(ControlMeshVerts[i + 1].Position[0], ControlMeshVerts[i + 1].Position[1], ControlMeshVerts[i + 1].Position[2]);
                u12 = uv[2 * (i + 1)];
                v12 = uv[2 * (i + 1) + 1];
            }
            if(i == 0) {
                s20 = new point(ControlMeshVerts[i].Position[0] + 1, ControlMeshVerts[i].Position[1] - 1, ControlMeshVerts[i].Position[2]);
                u20 = uv[2 * i] + 1;
                v20 = uv[2 * i + 1] - 1;
            } else {
                s20 = new point(ControlMeshVerts[i + 20].Position[0], ControlMeshVerts[i + 20].Position[1], ControlMeshVerts[i + 20].Position[2]);
                u20 = uv[2 * (i + 20)];
                v20 = uv[2 * (i + 20) + 1];
            }
            s21 = new point(ControlMeshVerts[i + 21].Position[0], ControlMeshVerts[i + 21].Position[1], ControlMeshVerts[i + 21].Position[2]);
            u21 = uv[2 * (i + 21)];
            v21 = uv[2 * (i + 21) + 1];
            if(i == 20) {
                s22 = new point(ControlMeshVerts[i].Position[0] + 1, ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
                u22 = uv[2 * i] + 1;
                v22 = uv[2 * i + 1] + 1;
            } else {
                s22 = new point(ControlMeshVerts[i + 22].Position[0], ControlMeshVerts[i + 22].Position[1], ControlMeshVerts[i + 22].Position[2]);
                u22 = uv[2 * (i + 22)];
                v22 = uv[2 * (i + 22) + 1];
            }

        } else if((i + 1) % 21 == 0 && i > 21 && i < 420) {
            //cout << "in top only" << endl;
            s00 = new point(ControlMeshVerts[i - 22].Position[0], ControlMeshVerts[i - 22].Position[1], ControlMeshVerts[i - 22].Position[2]);
            u00 = uv[2 * (i - 22)];
            v00 = uv[2 * (i - 22) + 1];
            s01 = new point(ControlMeshVerts[i - 21].Position[0], ControlMeshVerts[i - 21].Position[1], ControlMeshVerts[i - 21].Position[2]);
            u01 = uv[2 * (i - 21)];
            v01 = uv[2 * (i - 21) + 1];
            s02 = new point(ControlMeshVerts[i].Position[0] - 1, ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
            u02 = uv[2 * i] - 1;
            v02 = uv[2 * i + 1] - 1;
            s10 = new point(ControlMeshVerts[i - 1].Position[0], ControlMeshVerts[i - 1].Position[1], ControlMeshVerts[i - 1].Position[2]);
            u10 = uv[2 * (i - 1)];
            v10 = uv[2 * (i - 1) + 1];
            s11 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1], ControlMeshVerts[i].Position[2]);
            u11 = uv[2 * i];
            v11 = uv[2 * i + 1];
            s12 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
            u12 = uv[2 * i];
            v12 = uv[2 * i + 1] + 1;
            s20 = new point(ControlMeshVerts[i + 20].Position[0], ControlMeshVerts[i + 20].Position[1], ControlMeshVerts[i + 20].Position[2]);
            u20 = uv[2 * (i + 20)];
            v20 = uv[2 * (i + 20) + 1];
            s21 = new point(ControlMeshVerts[i + 21].Position[0], ControlMeshVerts[i + 21].Position[1], ControlMeshVerts[i + 21].Position[2]);
            u21 = uv[2 * (i + 21)];
            v21 = uv[2 * (i + 21) + 1];
            s22 = new point(ControlMeshVerts[i].Position[0] + 1, ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
            u22 = uv[2 * i] + 1;
            v22 = uv[2 * i + 1] + 1;
        } else if(i > 420 && ((i + 1) % 21 != 0 || i == 440)) {
            //cout << "in right only" << endl;
            s00 = new point(ControlMeshVerts[i - 22].Position[0], ControlMeshVerts[i - 22].Position[1], ControlMeshVerts[i - 22].Position[2]);
            u00 = uv[2 * (i - 22)];
            v00 = uv[2 * (i - 22) + 1];
            s01 = new point(ControlMeshVerts[i - 21].Position[0], ControlMeshVerts[i - 21].Position[1], ControlMeshVerts[i - 21].Position[2]);
            u01 = uv[2 * (i - 21)];
            v01 = uv[2 * (i - 21) + 1];
            if(i == 440) {
                s02 = new point(ControlMeshVerts[i].Position[0] - 1, ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
                u02 = uv[2 * i] - 1;
                v02 = uv[2 * i + 1] + 1;
            } else {
                s02 = new point(ControlMeshVerts[i - 20].Position[0], ControlMeshVerts[i - 20].Position[1], ControlMeshVerts[i - 20].Position[2]);
                u02 = uv[2 * (i - 20)];
                v02 = uv[2 * (i - 20) + 1];
            }
            s10 = new point(ControlMeshVerts[i - 1].Position[0], ControlMeshVerts[i - 1].Position[1], ControlMeshVerts[i - 1].Position[2]);
            u10 = uv[2 * (i - 1)];
            v10 = uv[2 * (i - 1) + 1];
            s11 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1], ControlMeshVerts[i].Position[2]);
            u11 = uv[2 * i];
            v11 = uv[2 * i + 1];
            if(i == 440) {
                s12 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
                u12 = uv[2 * i];
                v12 = uv[2 * i + 1] + 1;
            } else {
                s12 = new point(ControlMeshVerts[i + 1].Position[0], ControlMeshVerts[i + 1].Position[1], ControlMeshVerts[i + 1].Position[2]);
                u12 = uv[2 * (i + 1)];
                v12 = uv[2 * (i + 1) + 1];
            }
            s20 = new point(ControlMeshVerts[i].Position[0] + 1, ControlMeshVerts[i].Position[1] - 1, ControlMeshVerts[i].Position[2]);
            u20 = uv[2 * i] + 1;
            v20 = uv[2 * i + 1] - 1;
            s21 = new point(ControlMeshVerts[i].Position[0] + 1, ControlMeshVerts[i].Position[1], ControlMeshVerts[i].Position[2]);
            u21 = uv[2 * i] + 1;
            v21 = uv[2 * i + 1];
            s22 = new point(ControlMeshVerts[i].Position[0] + 1, ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
            u22 = uv[2 * i] + 1;
            v22 = uv[2 * i + 1] + 1;
        } else if(i % 21 == 0 && i >= 21 && i <= 420) {
            //cout << "in bottom only" << endl;
            s00 = new point(ControlMeshVerts[i].Position[0] - 1, ControlMeshVerts[i].Position[1] - 1, ControlMeshVerts[i].Position[2]);
            u00 = uv[2 * i] - 1;
            v00 = uv[2 * i + 1] - 1;
            s01 = new point(ControlMeshVerts[i - 21].Position[0], ControlMeshVerts[i - 21].Position[1], ControlMeshVerts[i - 21].Position[2]);
            u01 = uv[2 * (i - 21)];
            v01 = uv[2 * (i - 21) + 1];
            s02 = new point(ControlMeshVerts[i - 20].Position[0], ControlMeshVerts[i - 20].Position[1], ControlMeshVerts[i - 20].Position[2]);
            u02 = uv[2 * (i - 20)];
            v02 = uv[2 * (i - 20) + 1];
            s10 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1] - 1, ControlMeshVerts[i].Position[2]);
            u10 = uv[2 * i];
            v10 = uv[2 * i + 1] - 1;
            s11 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1], ControlMeshVerts[i].Position[2]);
            u11 = uv[2 * i];
            v11 = uv[2 * i + 1];
            s12 = new point(ControlMeshVerts[i + 1].Position[0], ControlMeshVerts[i + 1].Position[1], ControlMeshVerts[i + 1].Position[2]);
            u12 = uv[2 * (i + 1)];
            v12 = uv[2 * (i + 1) + 1];
            s20 = new point(ControlMeshVerts[i].Position[0] + 1, ControlMeshVerts[i].Position[1] - 1, ControlMeshVerts[i].Position[2]);
            u20 = uv[2 * i] + 1;
            v20 = uv[2 * i + 1] - 1;
            if(i == 420) {
                s21 = new point(ControlMeshVerts[i].Position[0] + 1, ControlMeshVerts[i].Position[1], ControlMeshVerts[i].Position[2]);
                u21 = uv[2 * i] + 1;
                v21 = uv[2 * i + 1];
            } else {
                s21 = new point(ControlMeshVerts[i + 21].Position[0], ControlMeshVerts[i + 21].Position[1], ControlMeshVerts[i + 21].Position[2]);
                u21 = uv[2 * (i + 21)];
                v21 = uv[2 * (i + 21) + 1];
            }
            if(i == 420) {
                s22 = new point(ControlMeshVerts[i].Position[0] + 1, ControlMeshVerts[i].Position[1] + 1, ControlMeshVerts[i].Position[2]);
                u22 = uv[2 * i] + 1;
                v22 = uv[2 * i + 1] + 1;
            } else {
                s22 = new point(ControlMeshVerts[i + 22].Position[0], ControlMeshVerts[i + 22].Position[1], ControlMeshVerts[i + 22].Position[2]);
                u22 = uv[2 * (i + 22)];
                v22 = uv[2 * (i + 22) + 1];
            }
        } else {
            //cout << "in middle only" << endl;
            s00 = new point(ControlMeshVerts[i - 22].Position[0], ControlMeshVerts[i - 22].Position[1], ControlMeshVerts[i - 22].Position[2]);
            u00 = uv[2 * (i - 22)];
            v00 = uv[2 * (i - 22) + 1];
            s01 = new point(ControlMeshVerts[i - 21].Position[0], ControlMeshVerts[i - 21].Position[1], ControlMeshVerts[i - 21].Position[2]);
            u01 = uv[2 * (i - 21)];
            v01 = uv[2 * (i - 21) + 1];
            s02 = new point(ControlMeshVerts[i - 20].Position[0], ControlMeshVerts[i - 20].Position[1], ControlMeshVerts[i - 20].Position[2]);
            u02 = uv[2 * (i - 20)];
            v02 = uv[2 * (i - 20) + 1];
            s10 = new point(ControlMeshVerts[i - 1].Position[0], ControlMeshVerts[i - 1].Position[1], ControlMeshVerts[i - 1].Position[2]);
            u10 = uv[2 * (i - 1)];
            v10 = uv[2 * (i - 1) + 1];
            s11 = new point(ControlMeshVerts[i].Position[0], ControlMeshVerts[i].Position[1], ControlMeshVerts[i].Position[2]);
            u11 = uv[2 * i] - 1;
            v11 = uv[2 * i + 1] - 1;
            s12 = new point(ControlMeshVerts[i + 1].Position[0], ControlMeshVerts[i + 1].Position[1], ControlMeshVerts[i + 1].Position[2]);
            u12 = uv[2 * (i + 1)];
            v12 = uv[2 * (i + 1) + 1];
            s20 = new point(ControlMeshVerts[i + 20].Position[0], ControlMeshVerts[i + 20].Position[1], ControlMeshVerts[i + 20].Position[2]);
            u20 = uv[2 * (i + 20)];
            v20 = uv[2 * (i + 20) + 1];
            s21 = new point(ControlMeshVerts[i + 21].Position[0], ControlMeshVerts[i + 21].Position[1], ControlMeshVerts[i + 21].Position[2]);
            u21 = uv[2 * (i + 21)];
            v21 = uv[2 * (i + 21) + 1];
            s22 = new point(ControlMeshVerts[i + 22].Position[0], ControlMeshVerts[i + 22].Position[1], ControlMeshVerts[i + 22].Position[2]);
            u22 = uv[2 * (i + 22)];
            v22 = uv[2 * (i + 22) + 1];
        }

        point c00 = (*s11 * (float)(16.0f/36.0f)) + ((*s21 + *s12 + *s01 + *s10) * (float)(4.0f/36.0f)) + ((*s22 + *s02 + *s00 + *s20) * (float)(1.0f/36.0f));
        float nu00 = (u11 * (float)(16.0f/36.0f)) + ((u21 + u12 + u01 + u10) * (float)(4.0f/36.0f)) + ((u22 + u02 + u00 + u20) * (float)(1.0f/36.0f));
        float nv00 = (v11 * (float)(16.0f/36.0f)) + ((v21 + v12 + v01 + v10) * (float)(4.0f/36.0f)) + ((v22 + v02 + v00 + v20) * (float)(1.0f/36.0f));

        point c01 = (*s11 * (float)(8.0f/18.0f)) + ((*s01 + *s21) * (float)(2.0f/18.0f)) + (*s12 * (float)(4.0f/18.0f)) + ((*s22 + *s02) * (float)(1.0f/18.0f));
        float nu01 = (u11 * (float)(8.0f/18.0f)) + ((u01 + u21) * (float)(2.0f/18.0f)) + (u12 * (float)(4.0f/18.0f)) + ((u22 + u02) * (float)(1.0f/18.0f));
        float nv01 = (v11 * (float)(8.0f/18.0f)) + ((v01 + v21) * (float)(2.0f/18.0f)) + (v12 * (float)(4.0f/18.0f)) + ((v22 + v02) * (float)(1.0f/18.0f));

        point c02 = (*s12 * (float)(8.0f/18.0f)) + ((*s02 + *s22) * (float)(2.0f/18.0f)) + (*s11 * (float)(4.0f/18.0f)) + ((*s21 + *s01) * (float)(1.0f/18.0f));
        float nu02 = (u12 * (float)(8.0f/18.0f)) + ((u02 + u22) * (float)(2.0f/18.0f)) + (u11 * (float)(4.0f/18.0f)) + ((u21 + u01) * (float)(1.0f/18.0f));
        float nv02 = (v12 * (float)(8.0f/18.0f)) + ((v02 + v22) * (float)(2.0f/18.0f)) + (v11 * (float)(4.0f/18.0f)) + ((v21 + v01) * (float)(1.0f/18.0f));

        point c10 = (*s11 * (float)(8.0f/18.0f)) + ((*s10 + *s12) * (float)(2.0f/18.0f)) + (*s21 * (float)(4.0f/18.0f)) + ((*s22 + *s20) * (float)(1.0f/18.0f));
        float nu10 = (u11 * (float)(8.0f/18.0f)) + ((u10 + u12) * (float)(2.0f/18.0f)) + (u21 * (float)(4.0f/18.0f)) + ((u22 + u20) * (float)(1.0f/18.0f));
        float nv10 = (v11 * (float)(8.0f/18.0f)) + ((v10 + v12) * (float)(2.0f/18.0f)) + (v21 * (float)(4.0f/18.0f)) + ((v22 + v20) * (float)(1.0f/18.0f));

        point c11 = (*s11 * (float)(4.0f/9.0f)) + ((*s21 + *s12) * (float)(2.0f/9.0f)) + ((*s22) * (float)(1.0f/9.0f));
        float nu11 = (u11 * (float)(4.0f/9.0f)) + ((u21 + u12) * (float)(2.0f/9.0f)) + ((u22) * (float)(1.0f/9.0f));
        float nv11 = (v11 * (float)(4.0f/9.0f)) + ((v21 + v12) * (float)(2.0f/9.0f)) + ((v22) * (float)(1.0f/9.0f));

        point c12 = (*s12 * (float)(4.0f/9.0f)) + ((*s11 + *s22) * (float)(2.0f/9.0f)) + (*s21 * (float)(1.0f/9.0f));
        float nu12 = (u12 * (float)(4.0f/9.0f)) + ((u11 + u22) * (float)(2.0f/9.0f)) + (u21 * (float)(1.0f/9.0f));
        float nv12 = (v12 * (float)(4.0f/9.0f)) + ((v11 + v22) * (float)(2.0f/9.0f)) + (v21 * (float)(1.0f/9.0f));

        point c20 = (*s21 * (float)(8.0f/18.0f)) + ((*s20 + *s22) * (float)(2.0f/18.0f)) + (*s11 * (float)(4.0f/18.0f) + (*s12 + *s10) * (float)(1.0f/18.0f));
        float nu20 = (u21 * (float)(8.0f/18.0f)) + ((u20 + u22) * (float)(2.0f/18.0f)) + (u11 * (float)(4.0f/18.0f) + (u12 + u10) * (float)(1.0f/18.0f));
        float nv20 = (v21 * (float)(8.0f/18.0f)) + ((v20 + v22) * (float)(2.0f/18.0f)) + (v11 * (float)(4.0f/18.0f) + (v12 + v10) * (float)(1.0f/18.0f));

        point c21 = (*s21 * (float)(4.0f/9.0f)) + ((*s11 + *s22) * (float)(2.0f/9.0f)) + (*s12 * (float)(1.0f/9.0f));
        float nu21 = (u21 * (float)(4.0f/9.0f)) + ((u11 + u22) * (float)(2.0f/9.0f)) + (u12 * (float)(1.0f/9.0f));
        float nv21 = (v21 * (float)(4.0f/9.0f)) + ((v11 + v22) * (float)(2.0f/9.0f)) + (v12 * (float)(1.0f/9.0f));

        point c22 = (*s22 * (float)(4.0f/9.0f)) + ((*s11 + *s22) * (float)(2.0f/9.0f)) + (*s11 * (float)(1.0f/9.0f));
        float nu22 = (u22 * (float)(4.0f/9.0f)) + ((u11 + u22) * (float)(2.0f/9.0f)) + (u11 * (float)(1.0f/9.0f));
        float nv22 = (v22 * (float)(4.0f/9.0f)) + ((v11 + v22) * (float)(2.0f/9.0f)) + (v11 * (float)(1.0f/9.0f));

        //cout << "s11 " << s11->x << " " << s11->y << " " << s11->z << endl;
        //cout << "c00 " << c00.x << " " << c00.y << " " << c00.z << endl;

        cout << "index 3 * j " << 3 * j << endl;
        ControlMeshSubdivVerts[3 * j].Position[0] = c00.x;
        ControlMeshSubdivVerts[3 * j].Position[1] = c00.y;
        ControlMeshSubdivVerts[3 * j].Position[2] = c00.z - 0.3f;
        ControlMeshSubdivVerts[3 * j].Position[3] = 1.0f;
        ControlMeshSubdivVerts[3 * j].SetColor(colorRed);
        ControlMeshSubdivVerts[3 * j].SetNormal(controlMeshNormal);
        uvSubdiv[2 * (3 * j)] = nu00;
        uvSubdiv[2 * (3 * j) + 1] = nv00;

        if((i + 1) % 21 != 0) {
            cout << "index 3 * j +1 " << 3 * j + 1 << endl;
            ControlMeshSubdivVerts[3 * j + 1].Position[0] = c01.x;
            ControlMeshSubdivVerts[3 * j + 1].Position[1] = c01.y;
            ControlMeshSubdivVerts[3 * j + 1].Position[2] = c01.z - 0.3f;
            ControlMeshSubdivVerts[3 * j + 1].Position[3] = 1.0f;
            ControlMeshSubdivVerts[3 * j + 1].SetColor(colorRed);
            ControlMeshSubdivVerts[3 * j + 1].SetNormal(controlMeshNormal);
            uvSubdiv[2 * (3 * j + 1)] = nu01;
            uvSubdiv[2 * (3 * j + 1) + 1] = nv01;

            cout << "index 3 * j + 2 " << 3 * j + 2 << endl;
            ControlMeshSubdivVerts[3 * j + 2].Position[0] = c02.x;
            ControlMeshSubdivVerts[3 * j + 2].Position[1] = c02.y;
            ControlMeshSubdivVerts[3 * j + 2].Position[2] = c02.z - 0.3f;
            ControlMeshSubdivVerts[3 * j + 2].Position[3] = 1.0f;
            ControlMeshSubdivVerts[3 * j + 2].SetColor(colorRed);
            ControlMeshSubdivVerts[3 * j + 2].SetNormal(controlMeshNormal);
            uvSubdiv[2 * (3 * j + 2)] = nu02;
            uvSubdiv[2 * (3 * j + 2) + 1] = nv02;
        }

        if(i < 420) {
            cout << "index 3 * j + 61 " << 3 * j + 61 << endl;
            ControlMeshSubdivVerts[3 * j + 61].Position[0] = c10.x;
            ControlMeshSubdivVerts[3 * j + 61].Position[1] = c10.y;
            ControlMeshSubdivVerts[3 * j + 61].Position[2] = c10.z - 0.3f;
            ControlMeshSubdivVerts[3 * j + 61].Position[3] = 1.0f;
            ControlMeshSubdivVerts[3 * j + 61].SetColor(colorRed);
            ControlMeshSubdivVerts[3 * j + 61].SetNormal(controlMeshNormal);
            uvSubdiv[2 * (3 * j + 61)] = nu10;
            uvSubdiv[2 * (3 * j + 61) + 1] = nv10;

            if((i + 1) % 21 != 0) {
                cout << "index 3 * j + 62 " << 3 * j + 62<< endl;
                ControlMeshSubdivVerts[3 * j + 62].Position[0] = c11.x;
                ControlMeshSubdivVerts[3 * j + 62].Position[1] = c11.y;
                ControlMeshSubdivVerts[3 * j + 62].Position[2] = c11.z - 0.3f;
                ControlMeshSubdivVerts[3 * j + 62].Position[3] = 1.0f;
                ControlMeshSubdivVerts[3 * j + 62].SetColor(colorRed);
                ControlMeshSubdivVerts[3 * j + 62].SetNormal(controlMeshNormal);
                uvSubdiv[2 * (3 * j + 62)] = nu11;
                uvSubdiv[2 * (3 * j + 62) + 1] = nv11;

                cout << "index 3 * j + 63 " << 3 * j + 63 << endl;
                ControlMeshSubdivVerts[3 * j + 63].Position[0] = c12.x;
                ControlMeshSubdivVerts[3 * j + 63].Position[1] = c12.y;
                ControlMeshSubdivVerts[3 * j + 63].Position[2] = c12.z - 0.3f;
                ControlMeshSubdivVerts[3 * j + 63].Position[3] = 1.0f;
                ControlMeshSubdivVerts[3 * j + 63].SetColor(colorRed);
                ControlMeshSubdivVerts[3 * j + 63].SetNormal(controlMeshNormal);
                uvSubdiv[2 * (3 * j + 63)] = nu12;
                uvSubdiv[2 * (3 * j + 63) + 1] = nv12;
            }

            cout << "index 3 * j + 122 " << 3 * j + 122 << endl;
            ControlMeshSubdivVerts[3 * j + 122].Position[0] = c20.x;
            ControlMeshSubdivVerts[3 * j + 122].Position[1] = c20.y;
            ControlMeshSubdivVerts[3 * j + 122].Position[2] = c20.z - 0.3f;
            ControlMeshSubdivVerts[3 * j + 122].Position[3] = 1.0f;
            ControlMeshSubdivVerts[3 * j + 122].SetColor(colorRed);
            ControlMeshSubdivVerts[3 * j + 122].SetNormal(controlMeshNormal);
            uvSubdiv[2 * (3 * j + 122)] = nu20;
            uvSubdiv[2 * (3 * j + 122) + 1] = nv20;

            if((i + 1) % 21 != 0) {
                cout << "index 3 * j + 123 " << 3 * j + 123 << endl;
                ControlMeshSubdivVerts[3 * j + 123].Position[0] = c21.x;
                ControlMeshSubdivVerts[3 * j + 123].Position[1] = c21.y;
                ControlMeshSubdivVerts[3 * j + 123].Position[2] = c21.z - 0.3f;
                ControlMeshSubdivVerts[3 * j + 123].Position[3] = 1.0f;
                ControlMeshSubdivVerts[3 * j + 123].SetColor(colorRed);
                ControlMeshSubdivVerts[3 * j + 123].SetNormal(controlMeshNormal);
                uvSubdiv[2 * (3 * j + 123)] = nu21;
                uvSubdiv[2 * (3 * j + 123) + 1] = nv21;

                cout << "index 3 * j + 124 " << 3 * j + 124 << endl;
                ControlMeshSubdivVerts[3 * j + 124].Position[0] = c22.x;
                ControlMeshSubdivVerts[3 * j + 124].Position[1] = c22.y;
                ControlMeshSubdivVerts[3 * j + 124].Position[2] = c22.z - 0.3f;
                ControlMeshSubdivVerts[3 * j + 124].Position[3] = 1.0f;
                ControlMeshSubdivVerts[3 * j + 124].SetColor(colorRed);
                ControlMeshSubdivVerts[3 * j + 124].SetNormal(controlMeshNormal);
                uvSubdiv[2 * (3 * j + 124)] = nu22;
                uvSubdiv[2 * (3 * j + 124) + 1] = nv22;
            }
        }

        if(i != 0 && (i + 1) % 21 == 0) {
            cout << "i " << i << " j " << j << endl;
            j = j + 41;
        } else {
            j++;
        }
    }

    for(int i = 0; i < 3721; i++) {
        //cout << "ControlMeshsubdivVerts[" << i << "] " << ControlMeshSubdivVerts[i].Position[0] << " " << ControlMeshSubdivVerts[i].Position[1] << " " << ControlMeshSubdivVerts[i].Position[2] << endl;
        if((i + 1) % 61 != 0 && i < 3660 && i != 3720) {
            ControlMeshSubdivIdcs[4 * i] = i;
            ControlMeshSubdivIdcs[4 * i + 1] = i + 1;
        } else {
            ControlMeshSubdivIdcs[4 * i] = i;
            ControlMeshSubdivIdcs[4 * i + 1] = i;
        }
        if(i < 3660) {
            ControlMeshSubdivIdcs[4 * i + 2] = i;
            ControlMeshSubdivIdcs[4 * i + 3] = i + 61;
        } else if(i != 3720) {
            ControlMeshSubdivIdcs[4 * i + 2] = i;
            ControlMeshSubdivIdcs[4 * i + 3] = i + 1;
        }

        if(i == 0 || (i + 1) % 61 != 0 && i < 3660) {
            ControlMeshSubdivIdcsForTex[6 * i] = i;
            ControlMeshSubdivIdcsForTex[6 * i + 1] = i + 1;
            ControlMeshSubdivIdcsForTex[6 * i + 2] = i + 62;
            ControlMeshSubdivIdcsForTex[6 * i + 3] = i + 62;
            ControlMeshSubdivIdcsForTex[6 * i + 4] = i + 61;
            ControlMeshSubdivIdcsForTex[6 * i + 5] = i;
        }

        /*uvSubdiv[2 * i] = (ControlMeshSubdivVerts[i].Position[0] + 10) / 60;
        uvSubdiv[2 * i + 1] = (ControlMeshSubdivVerts[i].Position[1]) / 60;*/
    }

    for(int i = 0; i < 7441; i+=2) {
        cout << "u " << uvSubdiv[i] << " v " << uvSubdiv[i+1] << endl;
    }

    VertexBufferSize[5] = sizeof(ControlMeshSubdivVerts);
    IndexBufferSize[5] = sizeof(ControlMeshSubdivIdcs);
    createVAOs(ControlMeshSubdivVerts, ControlMeshSubdivIdcs, 5);

    VertexBufferSize[6] = sizeof(ControlMeshSubdivVerts);
    IndexBufferSize[6] = sizeof(ControlMeshSubdivIdcsForTex);
    createVAOsForTex(ControlMeshSubdivVerts, ControlMeshSubdivIdcsForTex, 6);
}

void rotateRestGrid() {
    for(int i = 0; i < 189; i++) {
        ControlMeshVerts[i].Position[0] = -3.0f;
        if(i < 21) {
            ControlMeshVerts[i].Position[2] += 9;
        } else if(i >= 21 && i < 42) {
            ControlMeshVerts[i].Position[2] += 8;
        } else if(i >= 42 && i < 63) {
            ControlMeshVerts[i].Position[2] += 7;
        } else if(i >= 63 && i < 84) {
            ControlMeshVerts[i].Position[2] += 6;
        } else if(i >= 84 && i < 105) {
            ControlMeshVerts[i].Position[2] += 5;
        } else if(i >= 105 && i < 126) {
            ControlMeshVerts[i].Position[2] += 4;
        } else if(i >= 126 && i < 147) {
            ControlMeshVerts[i].Position[2] += 3;
        } else if(i >= 147 && i < 168) {
            ControlMeshVerts[i].Position[2] += 2;
        } else if(i >= 168 && i < 189) {
            ControlMeshVerts[i].Position[2] += 1;
        }
    }

    automate_ray(0, 189, 1, 0, 0, -1.0f);

    for(int i = 336; i < 441; i++) {
        if(i >= 336 && i < 357) {
            ControlMeshVerts[i].Position[2] += 1;
        } else if(i >= 357 && i < 378) {
            ControlMeshVerts[i].Position[2] += 2;
        } else if(i >= 378 && i < 399) {
            ControlMeshVerts[i].Position[2] += 3;
        } else if(i >= 399 && i < 420) {
            ControlMeshVerts[i].Position[2] += 4;
        } else if(i >= 420 && i < 441) {
            ControlMeshVerts[i].Position[2] += 5;
        }
    }

    automate_ray(336, 441, -1, 0, 0, 4.0f);

    for(int i = 15; i < 21; i++) {
        for(int j = 0; j < 21; j++) {
            ControlMeshVerts[21 * j + i].Position[2] += i - 15;
            if(21 * j + i + 1 < 440)
                automate_ray(21 * j + i, 21 * j + i + 1, 0, -1, 0, 14.0f);
        }
    }

    VertexBufferSize[2] = sizeof(ControlMeshVerts);
    IndexBufferSize[2] = sizeof(ControlMeshIdcs);
    createVAOs(ControlMeshVerts, ControlMeshIdcs, 2);

    VertexBufferSize[3] = sizeof(ControlMeshVerts);
    IndexBufferSize[3] = sizeof(ControlMeshIdcsForTex);
    createVAOsForTex(ControlMeshVerts, ControlMeshIdcsForTex, 3);
}

int main(void)
{
	// initialize window
	int errorCode = initWindow();
	if (errorCode != 0)
		return errorCode;

	// initialize OpenGL pipeline
	initOpenGL();

	// For speed computation
	double lastTime = glfwGetTime();
	int nbFrames = 0;
	do {
		//// Measure speed
		//double currentTime = glfwGetTime();
		//nbFrames++;
		//if (currentTime - lastTime >= 1.0){ // If last prinf() was more than 1sec ago
		//	// printf and reset
		//	printf("%f ms/frame\n", 1000.0 / double(nbFrames));
		//	nbFrames = 0;
		//	lastTime += 1.0;
		//}
		
		if (animation){
			phi += 0.01;
			if (phi > 360)
				phi -= 360;
		}

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)) {
            move_vertex();
        }
		// DRAWING POINTS
		renderScene();


	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
	glfwWindowShouldClose(window) == 0);

	cleanup();

	return 0;
}
