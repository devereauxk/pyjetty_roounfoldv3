#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of MC track information
  and do jet-finding, and save response histograms.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import argparse

# Data analysis and plotting
import numpy as np
import ROOT
import yaml
import array
import math

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjtools
import ecorrel

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_io_emb
from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.alice_analysis.process.user.substructure import process_mc_base
from pyjetty.alice_analysis.process.base import thermal_generator
from pyjetty.mputils.csubtractor import CEventSubtractor

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

n_bins_truth = [20, 22, 7] # WARNING RooUnfold seg faults if too many bins used
# these are the truth level binnings
binnings_truth = [np.logspace(-5,0,n_bins_truth[0]+1), \
            np.logspace(-2.299,0,n_bins_truth[1]+1), \
            np.array([5, 10, 20, 40, 60, 80, 100, 150]).astype(float) ]
# slight difference for reco pt bin
n_bins = [20, 22, 6]
binnings = [np.logspace(-5,0,n_bins[0]+1), \
            np.logspace(-2.299,0,n_bins[1]+1), \
            np.array([10, 20, 40, 60, 80, 100, 150]).astype(float) ]

event_counter = 0
n_matches = 0
n_truth = 0
n_reco = 0

################################################################
class EEC_pair:
	def __init__(self, _index1, _index2, _weight, _r, _pt):
		self.index1 = _index1
		self.index2 = _index2
		self.weight = _weight
		self.r = _r
		self.pt = _pt

	def is_equal(self, pair2):
		return (self.index1 == pair2.index1 and self.index2 == pair2.index2) \
			or (self.index1 == pair2.index2 and self.index2 == pair2.index1)
	
	def __str__(self):
		return "EEC pair with (index1, index2, weight, RL, pt) = (" + \
			str(self.index1) + ", " + str(self.index2) + ", " + str(self.weight) + \
			", " + str(self.r) + ", " + str(self.pt) + ")"


################################################################
class ProcessMC_ENC(process_mc_base.ProcessMCBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessMC_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

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
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
    
    # define binnings
    # these are the truth level binnings
    n_bins = [20, 22, 7] # WARNING RooUnfold seg faults if too many bins used
    # these are the truth level binnings
    binnings = [np.logspace(-5,0,n_bins[0]+1), \
                np.logspace(-2.299,0,n_bins[1]+1), \
                np.array([5, 10, 20, 40, 60, 80, 100, 150]).astype(float) ]
    # slight difference for reco pt bin
    n_bins_reco = [20, 22, 6]
    binnings_reco = [np.logspace(-5,0,n_bins_reco[0]+1), \
                np.logspace(-2.299,0,n_bins_reco[1]+1), \
                np.array([10, 20, 40, 60, 80, 100, 150]).astype(float) ]
    # using 8E-3 as the lower rl bin: solve 10^x = 8*10^(-3)

    h3_reco = ROOT.TH3D("reco", "reco", n_bins_reco[0], binnings_reco[0], n_bins_reco[1], binnings_reco[1], n_bins_reco[2], binnings_reco[2])
    setattr(self, "reco", h3_reco)
    h3_gen = ROOT.TH3D("gen", "gen", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])
    setattr(self, "gen", h3_gen)

    h1_reco = ROOT.TH1D("reco1D", "reco1D", n_bins_reco[2], binnings_reco[2])
    setattr(self, "reco1D", h1_reco)
    h1_gen = ROOT.TH1D("gen1D", "gen1D", n_bins[2], binnings[2])
    setattr(self, "gen1D", h1_gen)

    response = ROOT.RooUnfoldResponse(h3_reco, h3_gen)
    setattr(self, "response", response)
    response1D = ROOT.RooUnfoldResponse(h1_reco, h1_gen)
    setattr(self, "response1D", response1D)

    # for purity correction
    name = 'reco_unmatched'
    h = ROOT.TH3D("reco_unmatched", "reco_unmatched", n_bins_reco[0], binnings_reco[0], n_bins_reco[1], binnings_reco[1], n_bins_reco[2], binnings_reco[2])
    setattr(self, name, h)

    # efficiency and purity check for 1D jet pT unfolding
    name = 'gen1D_unmatched'
    h = ROOT.TH1D("gen1D_unmatched", "gen1D_unmatched", n_bins[2], binnings[2])
    setattr(self, name, h)

    name = 'reco1D_unmatched'
    h = ROOT.TH1D("reco1D_unmatched", "reco1D_unmatched", n_bins_reco[2], binnings_reco[2])
    setattr(self, name, h)


  #---------------------------------------------------------------
  # Analyze jets of a given event.
  #---------------------------------------------------------------
  def analyze_jets(self, jets_det, jets_truth, jetR, rho = 0):

    global n_reco
    global n_matches
    global n_truth
    global event_counter

    myptselector = 40
  
    jets_det_selected = fj.vectorPJ()

    # kinda unnessesary check
    for jet in jets_det:

      if jet.perp() <= self.jetpt_min_det:
        continue

      jets_det_selected.push_back(jet)

      getattr(self, 'reco1D_unmatched').Fill(jet.perp(), self.pt_hat)
      if jet.perp() > myptselector : n_reco += 1

    if self.debug_level > 1:
      print('Number of det-level jets: {}'.format(len(jets_det_selected)))
      
    ############################## JET MATCHING ##############################
    # perform jet-matching, every det jet has a guaranteed truth jet match
    det_used = []
    for t_jet in jets_truth:
      candidates = []
      candidates_pt = []

      if t_jet.eta() < 0.9 - self.jetR:
        getattr(self, 'gen1D_unmatched').Fill(t_jet.perp(), self.pt_hat)
      
      if t_jet.perp() > myptselector : n_truth += 1

      for i in range(jets_det_selected.size()):
        d_jet = jets_det_selected[i]

        if self.calculate_distance(t_jet, d_jet) < 0.2 and d_jet not in det_used:
          candidates.append(d_jet)
          candidates_pt.append(d_jet.perp())

      # if match found
      if len(candidates) > 0:
        winner_arg = np.argmin(np.abs(np.array(candidates_pt) - t_jet.perp()))
        det_match = candidates[winner_arg]
        det_match_pt = candidates_pt[winner_arg]
        det_used.append(det_match)

        getattr(self, "reco1D").Fill(det_match_pt, self.pt_hat)
        getattr(self, "gen1D").Fill(t_jet.perp(), self.pt_hat)

        self.fill_matched_jet_histograms(det_match, t_jet, det_match_pt)
        if det_match.perp() > myptselector : n_matches += 1

      # if match not found, DONT DO ANYTHING, we had a whole conversation about this...

    """
    if event_counter % 1000 == 0 and n_truth > 0 and n_reco > 0:
      print("truth {} det {} matches {}, [ {}, {} ]".format(n_truth, n_reco, n_matches, n_matches / n_truth, n_matches / n_reco))
    
    event_counter += 1
    """
      
  def fill_matched_jet_histograms(self, det_jet, truth_jet, det_pt):
    # assumes det and truth parts are matched beforehand:
    # matching particles are given matching user_index s
    # if some det or truth part does not have a match, it is given a unique index
    # also assumes all jet and particle level cuts have been applied already

	  # truth level EEC pairs
    truth_pairs = self.get_EEC_pairs(truth_jet, ipoint=2)

    # det level EEC pairs
    det_pairs = self.get_EEC_pairs(det_jet, ipoint=2)

    ######### purity correction #########
    # calculate det EEC cross section irregardless if truth match exists

    for d_pair in det_pairs:
       getattr(self, "reco_unmatched").Fill(d_pair.weight, d_pair.r, d_pair.pt, self.pt_hat)

    ########################## TTree output generation #########################
	  # composite of truth and smeared pairs, fill the TTree preprocessed
    dummyval = -9999

    # pair mathcing
    for t_pair in truth_pairs:

      getattr(self, "gen").Fill(t_pair.weight, t_pair.r, t_pair.pt, self.pt_hat)

      match_found = False
      for d_pair in det_pairs:

        if d_pair.is_equal(t_pair):

          getattr(self, "reco").Fill(d_pair.weight, d_pair.r, d_pair.pt, self.pt_hat)
          getattr(self, "response").Fill(d_pair.weight, d_pair.r, d_pair.pt, t_pair.weight, t_pair.r, t_pair.pt, self.pt_hat)

          match_found = True
          break

      if not match_found:

        getattr(self, "response").Miss(t_pair.weight, t_pair.r, t_pair.pt, self.pt_hat)
      


  def get_EEC_pairs(self, jet, ipoint=2):
    pairs = []

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
    EEC_indicies1 = EEC_cb.indices1() # contains list of 1st track in the pair (index should be based on the indices in c_select)
    EEC_indicies2 = EEC_cb.indices2() # contains list of 2nd track in the pair

    for i in range(len(EEC_rs)):
      event_index1 = _v[EEC_indicies1[i]].user_index()
      event_index2 = _v[EEC_indicies2[i]].user_index()
      pairs.append(EEC_pair(event_index1, event_index2, EEC_weights[i], EEC_rs[i], jet_pt))

    return pairs


##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process MC')
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

  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  # perform analysis
  analysis = ProcessMC_ENC(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_mc()
