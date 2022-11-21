CXX=g++
PROJECT=ADMULT
RM=rm -f
CFLAGS=`pkg-config --cflags opencv4`
LIBS=`pkg-config --libs opencv4`

SRCS=testeVideoModuloSMT.cpp testeVideoModulo02Lib.cpp
OBJS=$(subst .cpp,.o,$(SRCS))
 

ADMULT : testeVideoModuloSMT.o testeVideoModulo02Lib.o
	$(CXX) testeVideoModuloSMT.o testeVideoModulo02Lib.o $(CFLAGS) $(LIBS) -g -o ADMULT

testeVideoModuloSMT.o : testeVideoModuloSMT.cpp testeVideoModulo02Lib.h
	$(CXX) -c testeVideoModuloSMT.cpp  $(CFLAGS) $(LIBS)

testeVideoModulo02Lib.o : testeVideoModulo02Lib.cpp testeVideoModulo02Lib.h
	$(CXX) -c testeVideoModulo02Lib.cpp  $(CFLAGS) $(LIBS)
	
# %.o: %.cpp $(DEPS)
# 	$(CXX) -c -o $@ $<  $(LIBS) $(CFLAGS)

# $(PROJECT): $(OBJ)
# 	$(CXX) -o $@ $^  $(LIBS) $(CFLAGS)

clean:
	$(RM) $(OBJS)
 

# cmake -D CMAKE_BUILD_TYPE=RELEASE \
#     -D CMAKE_INSTALL_PREFIX=/home/luiz.tavares/.local \
#     -D INSTALL_C_EXAMPLES=ON \
#     -D INSTALL_PYTHON_EXAMPLES=ON \
#     -D OPENCV_GENERATE_PKGCONFIG=ON \
#     -D OPENCV_EXTRA_MODULES_PATH=/home/luiz.tavares/Softwares/opencv_build/opencv_contrib/modules \
#     -D OPENCV_ENABLE_NONFREE=ON \
#     -D BUILD_EXAMPLES=ON \
#     -D_GLIBCXX_USE_CXX11_ABI=0 ..