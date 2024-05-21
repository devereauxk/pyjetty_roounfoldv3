#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of track information
  and do jet-finding, and save basic histograms.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import argparse
import sys

# Data analysis and plotting
import ROOT
import yaml
import numpy as np
import array 
import math

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import ecorrel

# Base class
from pyjetty.alice_analysis.process.user.substructure import process_data_base

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class ProcessData_ENC(process_data_base.ProcessDataBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # print(sys.path)
    # print(sys.modules)

    # Initialize base class
    super(ProcessData_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    self.observable = self.observable_list[0]


  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
        
    n_bins_reco = [20, 22, 6]
    binnings_reco = [np.logspace(-5,0,n_bins_reco[0]+1), \
                np.logspace(-2.299,0,n_bins_reco[1]+1), \
                np.array([10, 20, 40, 60, 80, 100, 150]).astype(float) ]

    h3_raw = ROOT.TH3D("raw", "raw", n_bins_reco[0], binnings_reco[0], n_bins_reco[1], binnings_reco[1], n_bins_reco[2], binnings_reco[2])
    setattr(self, "raw", h3_raw)
    h1_raw =  ROOT.TH1D("raw1D", "raw1D", n_bins_reco[2], binnings_reco[2])
    setattr(self, "raw1D", h1_raw)

    h2_raw_eec = ROOT.TH2D("raw_eec", "raw_eec", n_bins_reco[1], binnings_reco[1], n_bins_reco[2], binnings_reco[2])
    setattr(self, "raw_eec", h2_raw_eec)
    

  #---------------------------------------------------------------
  # Calculate pair distance of two fastjet particles
  #---------------------------------------------------------------
  def calculate_distance(self, p0, p1):   
    dphiabs = math.fabs(p0.phi() - p1.phi())
    dphi = dphiabs

    if dphiabs > math.pi:
      dphi = 2*math.pi - dphiabs

    deta = p0.eta() - p1.eta()
    return math.sqrt(deta*deta + dphi*dphi)
  
  #---------------------------------------------------------------
  # Analyze jets of a given event.
  #---------------------------------------------------------------
  def analyze_jets(self, jets, jetR, fj_particles, rho = 0):

    # assumed pp so no area or leading track cuts applied

    for jet in jets:

      if jet.perp() <= self.jetpt_min_det:
        continue

      ################# JETS PASSED, FILL HISTS #################

      getattr(self, "raw1D").Fill(jet.perp())

      # fill signal histograms
      # must be placed after embedded 60 GeV particle check (PbPb only)!
      self.fill_jet_tables(jet)
      

  def fill_jet_tables(self, jet, ipoint=2):
    jet_pt = jet.perp()

    #push constutents to a vector in python
    #reapply pt cut incase of ghosts
    _v = fj.vectorPJ()
    for c in jet.constituents():
      if c.perp() > 1:
         _v.push_back(c)

    # n-point correlator with all charged particles
    max_npoint = 2
    weight_power = 1
    dphi_cut = -9999
    deta_cut = -9999
    cb = ecorrel.CorrelatorBuilder(_v, jet_pt, max_npoint, weight_power, dphi_cut, deta_cut)

    EEC_cb = cb.correlator(ipoint)

    EEC_weights = EEC_cb.weights() # cb.correlator(npoint).weights() constains list of weights
    EEC_rs = EEC_cb.rs() # cb.correlator(npoint).rs() contains list of RL
    
    for i in range(len(EEC_rs)):
        getattr(self, "raw").Fill(EEC_weights[i], EEC_rs[i], jet_pt)
        getattr(self, "raw_eec").Fill(EEC_rs[i], jet_pt, EEC_weights[i])
        
          

##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process data')
  parser.add_argument('-f', '--inputFile', action='store',
                      type=str, metavar='inputFile',
                      default='AnalysisResults.root',
                      help='Path of ROOT file containing TTrees')
  parser.add_argument('-c', '--configFile', action='store',
                      type=str, metavar='configFile',
                      default='config/analysis_config.yaml',
                      help="Path of config file for analysis")
  parser.add_argument('-o', '--outputDir', action='store',
                      type=str, metavar='outputDir',
                      default='./TestOutput',
                      help='Output directory for output to be written to')
  
  # Parse the arguments
  args = parser.parse_args()
  
  print('Configuring...')
  print('inputFile: \'{0}\''.format(args.inputFile))
  print('configFile: \'{0}\''.format(args.configFile))
  print('ouputDir: \'{0}\"'.format(args.outputDir))
  print('----------------------------------------------------------------')
  
  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = ProcessData_ENC(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_data()
