#pragma once

#include "GL/freeglut.h"
#include "GL/glut.h"
#include "GpuImplementation.h"


static class OpenGlRenderer {
private:
    static GLuint pbo;
    static GpuImplementation* gpuData;

public:
    static void OpenGlRendererStart(int argc, char** argv, GpuImplementation* gpuData);
    static void draw();
    static void resize(int w, int h);
    static void keyPressed(unsigned char key, int x, int y);
    static void initOGL();
    static void closeWindow();
};
