#!/usr/bin/env bash

maum_root=${MAUM_ROOT}
if [ -z "${maum_root}" ]; then
  echo 'MAUM_ROOT is not defined!'
  exit 1
fi

test -d ${maum_root} || mkdir -p ${maum_root}

export LD_LIBRARY_PATH=${maum_root}/lib:$LD_LIBRARY_PATH

repo_root=$(pwd)
[ -n $(which nproc) ] && {
  NPROC=$(nproc)
} || {
  NPROC=$(cat /proc/cpuinfo | grep cores | wc -l)
}
echo "[brain-mrc-train]" repo_root: ${repo_root}, NPROC: ${NPROC}, MAUM_ROOT: ${MAUM_ROOT}

OS=
if [ -f /etc/lsb-release ]; then
  OS=ubuntu
elif [ -f /etc/centos-release ]; then
  OS=centos
elif [ -f /etc/redhat-release ]; then
  OS=centos
else
  . /etc/os-release
  OS=$NAME
  if [[ $OS == Amazon* ]]; then
    OS=ami
  else
    echo $OS
    echo "Illegal OS, use ubuntu or centos"
    exit 1
  fi
fi

if [ ! ${DOCKER_MAUM_BUILD} ]; then
  echo ${DOCKER_MAUM_BUILD}
fi

function get_requirements() {
  if [ "${OS}" = "ubuntu" ]; then
    sudo apt-get install \
      libarchive13 \
      libarchive-dev \
      libatlas-base-dev \
      libatlas-dev \
      build-essential \
      python-dev \
      python3-dev \
      make \
      cmake \
      automake \
      libtool \
      g++-4.8 \
      g++
  else
    sudo yum install -y gcc \
      gcc-c++ \
      libarchive-devel.x86_64 \
      atlas-devel.x86_64 \
      python-devel.x86_64 \
      glibc.x86_64 \
      lapack-devel.x86_64 \
      autoconf \
      automake \
      libtool \
      make \
      cmake \
      cmake3
  fi

# find python>3.5.0
  PY="$(command -v python)"
  RIGHT_VERSION="$(${PY} -c "import sys;print(True if sys.version_info > (3, 5, 0) else False)")"
  ${PY} -c "import sys;print(sys.version_info)"
  echo $RIGHT_VERSION
  if [[ $RIGHT_VERSION == 'True' ]]; then
    echo "python location: {$PY}"
  else
    PY="$(command -v python3)"
    RIGHT_VERSION="$(${PY} -c "import sys;print(True if sys.version_info > (3, 5, 0) else False)")"
    ${PY} -c "import sys;print(sys.version_info)"
    echo $RIGHT_VERSION
    if [[ $RIGHT_VERSION == 'True' ]]; then
      echo "python location: {$PY}"
    else
      PY="$(command -v python3.5)"
      RIGHT_VERSION="$(${PY} -c "import sys;print(True if sys.version_info > (3, 5, 0) else False)")"
      ${PY} -c "import sys;print(sys.version_info)"
      echo $RIGHT_VERSION
      if [[ $RIGHT_VERSION == 'True' ]]; then
        echo "python location: {$PY}"
      else
        echo "There's no python3.5 at this machine OR there is a python3 lower than 3.5.1."
        echo "Install python3.5 first. (Stop build.)"

        PYTHON_VER=3.5.2
        PREFIX=/usr/local
        
        sudo mkdir -p $PREFIX
        sudo yum -y install wget make gcc xz openssl-devel bzip2-devel ncurses-devel gdbm-devel xz-devel sqlite-devel readline-devel tk-devel
        sudo wget https://www.python.org/ftp/python/${PYTHON_VER}/Python-${PYTHON_VER}.tar.xz
        sudo tar xvf Python-${PYTHON_VER}.tar.xz 
        cd Python-${PYTHON_VER}
        ./configure --prefix=${PREFIX} --enable-shared
        sudo CPU_CORE=$(lscpu | awk '/^CPU\(s\):/ {print $NF}')
        sudo make -j${CPU_CORE}
        sudo make altinstall
        #sudo cd -
        #sudo echo "${PREFIX}/lib" > /etc/ld.so.conf.d/python3.conf
        sudo sh -c "echo '/usr/local/lib' >> /etc/ld.so.conf.d/python3.conf"
        sudo ldconfig
        #sudo echo "export PATH=${PREFIX}/bin:\$PATH" >> /etc/profile
        sudo sh -c "echo 'export PATH=${PREFIX}/bin:\$PATH' >> /etc/profile"
        source /etc/profile
        PY="$(command -v python3)"
      fi
    fi
  fi
  curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
  sudo ${PY} get-pip.py
  rm get-pip.py
  PIP="$(command -v pip3.5)"
  sudo ${PIP} install virtualenvwrapper
  sudo ${PIP} install --upgrade virtualenvwrapper
  VIRTUAL_DIR="${HOME}/.virtualenvs"
  mkdir ${VIRTUAL_DIR}

  if [ -e /usr/local/bin/virtualenvwrapper.sh ]; then
    VW_PATH="/usr/local/bin/virtualenvwrapper.sh"
  else
    if [ -e /usr/bin/virtualenvwrapper.sh ]; then
      VW_PATH="/usr/bin/virtualenvwrapper.sh"
    else
  	  echo "virtualenvwrapper.sh is not founded"
	  exit 1
    fi
  fi

  if [[ "$(echo $WORKON_HOME)" == "" ]]; then
    echo "Add virtual args to ~/.bashrc"
    echo export WORKON_HOME=${HOME}/.virtualenvs >> ${HOME}/.bashrc
    echo export VIRTUALENVWRAPPER_PYTHON=${PY} >> ${HOME}/.bashrc
    source ~/.bashrc
    echo source $VW_PATH >> ${HOME}/.bashrc
  else
    echo "Virtual args are already in ~/.bashrc... skip.."
  fi

  source $VW_PATH
  mkvirtualenv -p ${PY} proto_clf
  lsvirtualenv
  workon proto_clf
  pip install --upgrade pip
  # Install pytorch separately
  # pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
  pip install -r requirements.txt
  deactivate
 }

GLOB_BUILD_DIR=${HOME}/.ensemble-classifier-build
test -d ${GLOB_BUILD_DIR} || mkdir -p ${GLOB_BUILD_DIR}
sha1=$(git log -n 1 --pretty=format:%H ./build.sh)
echo "Last commit for build.sh: ${sha1}"

# not docker build && update commit
if [ -z "${DOCKER_MAUM_BUILD}" ] && [ ! -f ${GLOB_BUILD_DIR}/${sha1}.done ]; then
  get_requirements
  if [ "$?" = "0" ]; then
    touch ${GLOB_BUILD_DIR}/${sha1}.done
  fi
else
  echo " get_requirements had been done!"
fi

__CC=${CC}
__CXX=${CXX}
if [ "${OS}" = "centos" ]; then
  if [ -z ${__CC} ]; then
    echo CC not defined, use /usr/bin/gcc
    __CC=/usr/bin/gcc
  fi
  if [ -z ${__CXX} ]; then
    echo CXX not defined, use /usr/bin/g++
    __CXX=/usr/bin/g++
  fi

  CMAKE=/usr/bin/cmake3
else
  if [ -z ${__CC} ]; then
    echo CC not defined, use /usr/bin/gcc
    __CC=/usr/bin/gcc
  fi
  if [ -z ${__CXX} ]; then
    echo CXX not defined, use /usr/bin/g++
    __CXX=/usr/bin/g++
  fi
  CMAKE=/usr/bin/cmake
fi

GCC_VER=$(${__CC} -dumpversion)

build_base="build-debug" && [[ "${MAUM_BUILD_DEPLOY}" == "true" ]] && build_base="build-deploy-debug"
build_dir=${PWD}/${build_base}-${GCC_VER}

# 다른 프로젝트의 build.sh tar .... 의 명령을 실행할 때 build.sh clean-deploy를 한 적이 있을 경우
# CMakeCache.txt가 예전 MAUM_ROOT(deploy-XXXXX)를 바라보는 문제가 있으므로 빌드 디렉토리를 새로 만든다
function build_mrc_train() {
  if [ "$MAUM_BUILD_DEPLOY" = true ]; then
    test -d ${build_dir} && rm -rf ${build_dir}
  fi

  test -d ${build_dir} || mkdir -p ${build_dir}
  cd ${build_dir}

  ${CMAKE} \
    -DCMAKE_PREFIX_PATH=${maum_root} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=${__CC} \
    -DCMAKE_CXX_COMPILER=${__CXX} \
    -DCMAKE_INSTALL_PREFIX=${maum_root} ..

  if [ "$1" = "proto" ]; then
    (cd proto && make install -j${NPROC})
    (cd pysrc/proto && make install -j${NPROC})
  else
    if [ "${MAUM_BUILD_DEPLOY}" = "true" ]; then
      make install -j${NPROC}
    else
      make install
    fi
  fi
}

build_mrc_train
