#include <iostream>
#include "OpenGlRenderer.h"

GLuint OpenGlRenderer::pbo = 0;
GpuImplementation* OpenGlRenderer::gpuData = 0;

void OpenGlRenderer::OpenGlRendererStart(int argc, char** argv, GpuImplementation* gpuData) {
    OpenGlRenderer::gpuData = gpuData;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH);
    glutInitWindowSize(400, 400);
    glutCreateWindow("Canny Edge Display");
    glutDisplayFunc(OpenGlRenderer::draw);
    glutReshapeFunc(OpenGlRenderer::resize);
    glutKeyboardFunc(OpenGlRenderer::keyPressed);
    glutCloseFunc(OpenGlRenderer::closeWindow);

    initOGL();

    glutPostRedisplay();

    glutMainLoop();
}

void OpenGlRenderer::draw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //glClear(GL_COLOR_BUFFER_BIT);

    //glDisable(GL_LIGHTING);
    //glColor3f(1, 0, 0);
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
    //glEnable(GL_LIGHTING);

    glFlush();
}

void OpenGlRenderer::resize(int w, int h) {
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // glOrtho(-1,1,-1,1,-1,1);
    //glOrtho(-2.0f, 2.0f, -2.0f, 2.0f, -2.0f, 2.0f);
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
    case (27): {
        closeWindow();
        break;
    }
    }
}

void OpenGlRenderer::initOGL() { // Setup Shading Environment

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glShadeModel(GL_SMOOTH);


    // Enable features we want to use from OpenGL


    glEnable(GL_POLYGON_SMOOTH);
    glEnable(GL_LINE_SMOOTH);

    glEnable(GL_CULL_FACE);
    //glCullFace(GL_FRONT_AND_BACK);


    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glClearColor(0.45f, 0.45f, 0.45f, 1.0f);

	glGenTextures(1, &OpenGlRenderer::pbo);
    glBindTexture(GL_TEXTURE_2D, OpenGlRenderer::pbo);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, OpenGlRenderer::gpuData->imageWidth, OpenGlRenderer::gpuData->imageHeight, 0,
        GL_RED, GL_FLOAT,
        gpuData->h_outputGpu.data());


    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


}

void OpenGlRenderer::closeWindow() { glutLeaveMainLoop(); }
