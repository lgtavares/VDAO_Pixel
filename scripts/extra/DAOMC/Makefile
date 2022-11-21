CXX=g++
PROJECT=DAOMC
RM=rm -f
CFLAGS=`pkg-config --cflags opencv4`
LIBS=`pkg-config --libs opencv4`

SRCS=localizaDummy03All59SMT.cpp localizaDummy03Lib.cpp
OBJS=$(subst .cpp,.o,$(SRCS))
 

DAOMC : localizaDummy03All59SMT.o localizaDummy03Lib.o
	$(CXX) localizaDummy03All59SMT.o localizaDummy03Lib.o $(CFLAGS) $(LIBS) -g -o DAOMC

localizaDummy03All59SMT.o : localizaDummy03All59SMT.cpp localizaDummy03Lib.h
	$(CXX) -c localizaDummy03All59SMT.cpp  $(CFLAGS) $(LIBS)

localizaDummy03Lib.o : localizaDummy03Lib.cpp localizaDummy03Lib.h
	$(CXX) -c localizaDummy03Lib.cpp  $(CFLAGS) $(LIBS)
	
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