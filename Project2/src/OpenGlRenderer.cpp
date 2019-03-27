#include "OpenGlRenderer.h"
#include <iostream>

GLuint OpenGlRenderer::pbo = 0;
GpuImplementation* OpenGlRenderer::gpuData = 0;
float OpenGlRenderer::T1 = 0.4;
float OpenGlRenderer::T2 = 0.8;
GLvoid* OpenGlRenderer::pixels = 0;

void OpenGlRenderer::OpenGlRendererStart(int argc, char** argv, GpuImplementation* gpuData) {
    OpenGlRenderer::gpuData = gpuData;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(400, 400);
    glutCreateWindow("Canny Edge Display");
    glutDisplayFunc(OpenGlRenderer::draw);
    glutReshapeFunc(OpenGlRenderer::resize);
    glutKeyboardFunc(OpenGlRenderer::keyPressed);
    glutSpecialFunc(OpenGlRenderer::specialKeyPressed);
    glutCloseFunc(OpenGlRenderer::closeWindow);

    initOGL();

    glutPostRedisplay();

    glutMainLoop();
}

void OpenGlRenderer::draw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glClear(GL_COLOR_BUFFER_BIT);

    // glDisable(GL_LIGHTING);
    // glColor3f(1, 0, 0);
    glBindTexture(GL_TEXTURE_2D, OpenGlRenderer::pbo);
    glEnable(GL_TEXTURE_2D);


    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 1.0);
    glVertex3f(-1, -1, 0);

    glTexCoord2d(1.0, 1.0);
    glVertex3f(1, -1, 0);

    glTexCoord2d(1.0, 0.0);
    glVertex3f(1, 1, 0);

    glTexCoord2d(0.0, 0.0);
    glVertex3f(-1, 1, 0);
    glEnd();
    // glEnable(GL_LIGHTING);

    glFlush();
}

void OpenGlRenderer::resize(int w, int h) {
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // glOrtho(-1,1,-1,1,-1,1);
    // glOrtho(-2.0f, 2.0f, -2.0f, 2.0f, -2.0f, 2.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void OpenGlRenderer::keyPressed(unsigned char key, int x, int y) {
    printf("pressed normal key is %c\n", key);

    int mod = glutGetModifiers();
    if (mod != 0) {
        // ALT=4  SHIFT=1  CTRL=2
        switch (mod) {
        case 1:
            printf("SHIFT key %d\n", mod);
            break;
        case 2:
            printf("CTRL  key %d\n", mod);
            break;
        case 4:
            printf("ALT   key %d\n", mod);
            break;
            mod = 0;
        }
    }

    switch (key) {
    case (27): { // escape key
        closeWindow();
        break;
    }
    }
}

void OpenGlRenderer::specialKeyPressed(int key, int x, int y) {
    switch (key) {
    case (GLUT_KEY_DOWN): { 
        OpenGlRenderer::T2 += 0.05;
        OpenGlRenderer::loadImage();
        break;
    }
    case (GLUT_KEY_UP): { 
        OpenGlRenderer::T2 -= 0.05;
        if (OpenGlRenderer::T2 < 0) OpenGlRenderer::T2 = 0;
        OpenGlRenderer::loadImage();
        break;
    }
    case (GLUT_KEY_LEFT): {
        OpenGlRenderer::T1 += 0.05;
        OpenGlRenderer::loadImage();
        break;
    }
    case (GLUT_KEY_RIGHT): {
        OpenGlRenderer::T1 -= 0.05;
        if (OpenGlRenderer::T1 < 0) OpenGlRenderer::T1 = 0;
        OpenGlRenderer::loadImage();
        break;
    }
    }
    //glutPostRedisplay();
}

void OpenGlRenderer::initOGL() { // Setup Shading Environment

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glShadeModel(GL_SMOOTH);


    // Enable features we want to use from OpenGL


    glEnable(GL_POLYGON_SMOOTH);
    glEnable(GL_LINE_SMOOTH);

    glEnable(GL_CULL_FACE);
    // glCullFace(GL_FRONT_AND_BACK);


    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glClearColor(0.45f, 0.45f, 0.45f, 1.0f);

    glGenTextures(1, &OpenGlRenderer::pbo);
    glBindTexture(GL_TEXTURE_2D, OpenGlRenderer::pbo);


    int count = OpenGlRenderer::gpuData->imageWidth * OpenGlRenderer::gpuData->imageHeight;

    OpenGlRenderer::pixels = new float[count];
    memset(OpenGlRenderer::pixels, 0, count * sizeof(float));


    OpenGlRenderer::loadImage();


    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, OpenGlRenderer::gpuData->imageWidth, OpenGlRenderer::gpuData->imageHeight, 0,
    //    GL_RED, GL_FLOAT, OpenGlRenderer::pixels);


    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void OpenGlRenderer::closeWindow() { glutLeaveMainLoop(); }

void OpenGlRenderer::loadImage() {
    OpenGlRenderer::gpuData->executeWithouSave(T1, T2);
    OpenGlRenderer::pixels = OpenGlRenderer::gpuData->h_outputGpu.data();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, OpenGlRenderer::gpuData->imageWidth, OpenGlRenderer::gpuData->imageHeight, 0,
        GL_RED, GL_FLOAT, OpenGlRenderer::pixels);
    OpenGlRenderer::draw();
}
