################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/CPUImplementation.cpp \
../src/GpuImplementation.cpp \
../src/OpenGlRenderer.cpp \
../src/main.cpp 

OBJS += \
./src/CPUImplementation.o \
./src/GpuImplementation.o \
./src/OpenGlRenderer.o \
./src/main.o 

CPP_DEPS += \
./src/CPUImplementation.d \
./src/GpuImplementation.d \
./src/OpenGlRenderer.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -DOMPI_SKIP_MPICXX -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -I"/home/knodelmn/FaPra/gpuLab/Project/lib" -I/usr/include/hdf5/serial -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


