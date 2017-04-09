if [ ${TRAVIS_OS_NAME} == "linux" ]; then
  mkdir shadow_bin
  ln -s `which gcc-4.8` shadow_bin/gcc
  ln -s `which g++-4.8` shadow_bin/g++

  export PATH=$PWD/shadow_bin:$PATH
fi
