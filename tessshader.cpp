#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>

#include "tessshader.hpp"

GLuint LoadTessShaders(const char * tess_control_file_path,const char * tess_eval_file_path, const char * vertex_file_path,const char * fragment_file_path){
    // Create the shaders
    GLuint TessControlShaderID = glCreateShader(GL_TESS_CONTROL_SHADER);
    GLuint TessEvalShaderID = glCreateShader(GL_TESS_EVALUATION_SHADER);
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the Tessellation Control Shader code from the file
    std::string TessControlShaderCode;
    std::ifstream TessControlShaderStream(tess_control_file_path, std::ios::in);
    if(TessControlShaderStream.is_open()){
            std::string Line = "";
            while(getline(TessControlShaderStream, Line))
                    TessControlShaderCode += "\n" + Line;
            TessControlShaderStream.close();
    }else{
            printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", tess_control_file_path);
            getchar();
            return 0;
    }

    // Read the Tessellation Evaluation Shader code from the file
    std::string TessEvalShaderCode;
    std::ifstream TessEvalShaderStream(tess_eval_file_path, std::ios::in);
    if(TessEvalShaderStream.is_open()){
            std::string Line = "";
            while(getline(TessEvalShaderStream, Line))
                    TessEvalShaderCode += "\n" + Line;
            TessEvalShaderStream.close();
    }

    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
    if(VertexShaderStream.is_open()){
        std::string Line = "";
        while(getline(VertexShaderStream, Line))
            VertexShaderCode += "\n" + Line;
        VertexShaderStream.close();
    }else{
        printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
        getchar();
        return 0;
    }

    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
    if(FragmentShaderStream.is_open()){
        std::string Line = "";
        while(getline(FragmentShaderStream, Line))
            FragmentShaderCode += "\n" + Line;
        FragmentShaderStream.close();
    }

    GLint Result = GL_FALSE;
    int InfoLogLength;


    // Compile Tessellation Control Shader
    printf("Compiling shader : %s\n", tess_control_file_path);
    char const * TessControlSourcePointer = TessControlShaderCode.c_str();
    glShaderSource(TessControlShaderID, 1, &TessControlSourcePointer , NULL);
    glCompileShader(TessControlShaderID);

    // Check Tessellation Control Shader
    glGetShaderiv(TessControlShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(TessControlShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
            std::vector<char> TessControlShaderErrorMessage(InfoLogLength+1);
            glGetShaderInfoLog(TessControlShaderID, InfoLogLength, NULL, &TessControlShaderErrorMessage[0]);
            printf("%s\n", &TessControlShaderErrorMessage[0]);
    }



    // Compile Tessellation Evaluation Shader
    printf("Compiling shader : %s\n", tess_eval_file_path);
    char const * TessEvalSourcePointer = TessEvalShaderCode.c_str();
    glShaderSource(TessEvalShaderID, 1, &TessEvalSourcePointer , NULL);
    glCompileShader(TessEvalShaderID);

    // Check Tessellation Evaluation Shader
    glGetShaderiv(TessEvalShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(TessEvalShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
            std::vector<char> TessEvalShaderErrorMessage(InfoLogLength+1);
            glGetShaderInfoLog(TessEvalShaderID, InfoLogLength, NULL, &TessEvalShaderErrorMessage[0]);
            printf("%s\n", &TessEvalShaderErrorMessage[0]);
    }

    // Compile Vertex Shader
    printf("Compiling shader : %s\n", vertex_file_path);
    char const * VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
    glCompileShader(VertexShaderID);

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
    }



    // Compile Fragment Shader
    printf("Compiling shader : %s\n", fragment_file_path);
    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
    }

    // Link the program
    printf("Linking program\n");
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, TessControlShaderID);
    glAttachShader(ProgramID, TessEvalShaderID);
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
            std::vector<char> ProgramErrorMessage(InfoLogLength+1);
            glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
            printf("%s\n", &ProgramErrorMessage[0]);
    }


    glDetachShader(ProgramID, TessControlShaderID);
    glDetachShader(ProgramID, TessEvalShaderID);
    glDetachShader(ProgramID, VertexShaderID);
    glDetachShader(ProgramID, FragmentShaderID);

    glDeleteShader(TessControlShaderID);
    glDeleteShader(TessEvalShaderID);
    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}
