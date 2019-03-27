#pragma once

#include "GL/freeglut.h"
#include "GL/glut.h"
#include "GpuImplementation.h"


class OpenGlRenderer {
private:
    static GLuint pbo;
    static GpuImplementation* gpuData;
    static float T1;
    static float T2;
    static GLvoid* pixels;

public:
    static void OpenGlRendererStart(int argc, char** argv, GpuImplementation* gpuData);
    static void draw();
    static void resize(int w, int h);
    static void keyPressed(unsigned char key, int x, int y);
    static void specialKeyPressed(int key, int x, int y);
    static void initOGL();
    static void closeWindow();
    static void loadImage();
};
