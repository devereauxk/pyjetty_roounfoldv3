# pyjetty
### with RooUnfold v3 so you can do 3D unfolding easy

assuming that this is being done on perlmutter

adapted from https://github.com/alicernc/info/blob/main/heppy_at_perlmutter.md

## compile heppy with RooUnfold v3

```
workdir=/global/cfs/cdirs/alice/$USER/mypyjetty_roounfold
mkdir -p $workdir
cd $workdir
## module load python/3.11 - not needed we will use systems python3 - change 5-Oct-2023
python3 -m venv pyjettyenv_roounfold
source pyjettyenv_roounfold/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy tqdm pyyaml

# load some preinstalled packages
## module use /global/cfs/cdirs/alice/heppy_soft/yasp/software/modules - gsl not present - change 5-Oct-2023
module use /global/cfs/cdirs/alice/heppy_soft/05-11-2023/yasp/software/modules
module load cmake gsl root/6.28.00 HepMC2/2.06.11 LHAPDF6/6.5.3 pcre2/default swig/4.1.1 HepMC3/3.2.5

git clone https://github.com/matplo/heppy.git
./heppy/external/fastjet/build.sh
./heppy/external/pythia8/build.sh
./heppy/external/roounfold/build.sh --v3
# the v3 tag here ensures you get RooUnfold v3
./heppy/cpptools/build.sh

# note: use --clean when recompiling...
# like that: ./heppy/external/fastjet/build.sh --clean
# ...
```

## compile pyjetty

I tried to clone and compile with Wenqing's pyjetty here, but for some reason it disagrees. Since `ecorrel` files necessary for EEC unfolding, I just copy those over to Mateusz's pyjetty before compiling...
```
workdir=/global/cfs/cdirs/alice/$USER/mypyjetty
cd ${workdir}
module use ${workdir}/heppy/modules
module load heppy
# two lines below if new shell/terminal
module use /global/cfs/cdirs/alice/heppy_soft/05-11-2023/yasp/software/modules
module load cmake gsl root/6.28.00 HepMC2/2.06.11 LHAPDF6/6.5.3 pcre2/default swig/4.1.1 HepMC3/3.2.5
git clone git@github.com:matplo/pyjetty.git
./pyjetty/cpptools/build.sh --tglaubermc --tenngen
```

## ~/.bashrc

```
function pyjetty_w_roounfoldv3_load()
{
        workdir=/global/cfs/cdirs/alice/$USER/mypyjetty_roounfold
        source $workdir/pyjettyenv_roounfold/bin/activate
        module use /global/cfs/cdirs/alice/heppy_soft/05-11-2023/yasp/software/modules
        module load cmake gsl root/6.28.00 HepMC2/2.06.11 LHAPDF6/6.5.3 pcre2/default swig/4.1.1 HepMC3/3.2.5
        module use $workdir/pyjetty/modules
        module load pyjetty

        module list
}
export -f pyjetty_w_roounfoldv3_load
```

## testing the implementation

```
pyjetty_w_roounfoldv3_load
workdir=/global/cfs/cdirs/alice/$USER/mypyjetty_roounfold
cd $workdir
$PYJETTY_DIR/pyjetty/examples/pythia_gen_fastjet_lund_test.py
```

