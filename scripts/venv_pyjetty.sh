#!/bin/bash

savedir=${PWD}

function thisdir()
{
	SOURCE="${BASH_SOURCE[0]}"
	while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
	  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
	  SOURCE="$(readlink "$SOURCE")"
	  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
	done
	DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
	echo ${DIR}
}
THISD=$(thisdir)

cd ${THISD}


# echo "[current dir:] ${THISD}"
module use ${THISD}/../modules
module load pyjetty

echo "[i] Using HEPPY_DIR $HEPPY_DIR"
#venvdir=${THISD}/../venvheppy
venvdir=${HEPPY_DIR}/venvheppy

cmnd=$@

first_run=""
if [ ! -d ${venvdir} ]; then
	echo "[i] creating venv"
	python3 -m venv ${venvdir}
	first_run="yes"
fi

tmpfile=$(mktemp)	
if [ -d ${venvdir} ]; then
	echo "export PS1=\"\e[32;1m[\u\e[31;1m@\h\e[32;1m]\e[34;1m\w\e[0m\n> \"" > ${tmpfile}
	if [ -e "$HOME/.bashrc" ]; then
		echo "source $HOME/.bashrc" >> ${tmpfile}
	fi
	echo "source ${venvdir}/bin/activate" >> ${tmpfile}
	if [ "x${first_run}" == "xyes" ]; then
		echo "[i] first run? ${first_run}"
		echo "python -m pip install --upgrade pip" >> ${tmpfile}
		echo "python -m pip install pyyaml numpy" >> ${tmpfile}
	fi
	if [ ! -e "${THISD}/.venvstartup.sh" ]; then
		echo "module use ${THISD}/../modules" > ${THISD}/.venvstartup.sh
		echo "module avail" >> ${THISD}/.venvstartup.sh
		echo "module load pyjetty" >> ${THISD}/.venvstartup.sh
		echo "module list" >> ${THISD}/.venvstartup.sh
	fi
	echo "source ${THISD}/.venvstartup.sh" >> ${tmpfile}
	echo "cd ${savedir}" >> ${tmpfile}
	if [ -z "${cmnd}" ]; then
		/bin/bash --init-file ${tmpfile} -i
	else
		echo "[i] exec ${tmpfile}" >&2
		echo "${cmnd}" >> ${tmpfile}
		chmod +x ${tmpfile}
		${tmpfile}
	fi
	#-s < .activate
fi

cd ${savedir}
