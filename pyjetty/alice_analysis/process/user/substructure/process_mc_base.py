#!/usr/bin/env python3

"""
Base class to read a ROOT TTree of track information
and do jet-finding, and save basic histograms.
  
To use this class, the following should be done:

  - Implement a user analysis class inheriting from this one, such as in user/james/process_mc_XX.py
    You should implement the following functions:
      - initialize_user_output_objects_R()
      - fill_observable_histograms()
      - fill_matched_jet_histograms()
    
  - You should include the following histograms:
      - Response matrix: hResponse_JetPt_[obs]_R[R]_[subobs]_[grooming setting]
      - Residual distribution: hResidual_JetPt_[obs]_R[R]_[subobs]_[grooming setting]

  - You also should modify observable-specific functions at the top of common_utils.py
  
Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import time

# Data analysis and plotting
import pandas
import numpy as np
import array
import ROOT
import yaml
import math
import sys

# Fastjet via python (from external library heppy)
import fastjet as fj

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_io_emb
from pyjetty.alice_analysis.process.base import process_base
from pyjetty.alice_analysis.process.base import thermal_generator

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class ProcessMCBase(process_base.ProcessBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessMCBase, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

    # find pt_hat for set of events in input_file, assumes all events in input_file are in the same pt_hat bin
    self.pt_hat_bin = int(input_file.split('/')[len(input_file.split('/'))-4]) # depends on exact format of input_file name
    with open("/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC18b8/scaleFactors.yaml", 'r') as stream:
        pt_hat_yaml = yaml.safe_load(stream)
    self.pt_hat = pt_hat_yaml[self.pt_hat_bin]
    print("pt hat bin : " + str(self.pt_hat_bin))
    print("pt hat weight : " + str(self.pt_hat))
    
    # Initialize configuration
    self.initialize_config()
    
  #---------------------------------------------------------------
  # Initialize config file into class members
  #---------------------------------------------------------------
  def initialize_config(self):
    
    # Call base class initialization
    process_base.ProcessBase.initialize_config(self)
    
    # Read config file
    with open(self.config_file, 'r') as stream:
      config = yaml.safe_load(stream)
      
    self.fast_simulation = config['fast_simulation']
    if self.fast_simulation == True:
      if 'ENC_fastsim' in config:
          self.ENC_fastsim = config['ENC_fastsim']
      else:
          self.ENC_fastsim = False
    else: # if not fast simulation, set ENC_fastsim flag to False
      self.ENC_fastsim = False    
    if 'jetscape' in config:
        self.jetscape = config['jetscape']
    else:
        self.jetscape = False
    if 'event_plane_angle' in config:
      self.event_plane_range = config['event_plane_angle']
    else:
      self.event_plane_range = None
    if 'matching_systematic' in config:
      self.matching_systematic = config['matching_systematic']
    else:
      self.matching_systematic = False
    self.skip_deltapt_RC_histograms = True
    self.fill_RM_histograms = True
    
    self.reject_tracks_fraction = config['reject_tracks_fraction']
    if 'mc_fraction_threshold' in config:
      self.mc_fraction_threshold = config['mc_fraction_threshold']

    self.is_pp = config['is_pp']

    if not self.is_pp:
      self.emb_file_list = config['emb_file_list']

    if 'jetpt_min_det' in config:
      self.jetpt_min_det = config['jetpt_min_det']
    else:
      self.jetpt_min_det = 10

    if 'jetpt_min_truth' in config:
      self.jetpt_min_truth = config['jetpt_min_truth']
    else:
      self.jetpt_min_truth = 5

    if 'jetpt_min_det_subtracted' in config:
      self.jetpt_min_det_subtracted = config['jetpt_min_det_subtracted']
    else:
      self.jetpt_min_det_subtracted = 10

    if 'jetR' in config:
      self.jetR = config['jetR']
    else:
      self.jetR = 0.4
        
    if 'thermal_model' in config:
      self.thermal_model = True
      beta = config['thermal_model']['beta']
      N_avg = config['thermal_model']['N_avg']
      sigma_N = config['thermal_model']['sigma_N']
      self.thermal_generator = thermal_generator.ThermalGenerator(N_avg, sigma_N, beta)
    else:
      self.thermal_model = False

    # Create dictionaries to store grooming settings and observable settings for each observable
    # Each dictionary entry stores a list of subconfiguration parameters
    #   The observable list stores the observable setting, e.g. subjetR
    #   The grooming list stores a list of grooming settings {'sd': [zcut, beta]} or {'dg': [a]}
    self.observable_list = config['process_observables']
    self.obs_settings = {}
    self.obs_grooming_settings = {}
    for observable in self.observable_list:
    
      obs_config_dict = config[observable]
      obs_config_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]
      
      obs_subconfig_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]
      self.obs_settings[observable] = self.utils.obs_settings(observable, obs_config_dict, obs_subconfig_list)
      self.obs_grooming_settings[observable] = self.utils.grooming_settings(obs_config_dict)

    # KD: assume one observable setting/label
    observable = self.observable_list[0]
    self.obs_setting = 0.15 # hard coded
    grooming_setting = self.obs_grooming_settings[observable][0]
    self.obs_label = self.utils.obs_label(self.obs_setting, grooming_setting)
  
  #---------------------------------------------------------------
  # Main processing function
  #---------------------------------------------------------------
  def process_mc(self):
    
    self.start_time = time.time()
    
    # Use IO helper class to convert detector-level ROOT TTree into
    # a SeriesGroupBy object of fastjet particles per event
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    if self.fast_simulation:
      tree_dir = ''
    else:
      tree_dir = 'PWGHF_TreeCreator'
    io_det = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                  track_tree_name='tree_Particle', use_ev_id_ext=False,
                                  is_jetscape=self.jetscape, event_plane_range=self.event_plane_range, is_ENC=self.ENC_fastsim, is_det_level=True)
    df_fjparticles_det = io_det.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
    self.nEvents_det = len(df_fjparticles_det.index)
    self.nTracks_det = len(io_det.track_df.index)
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    # If jetscape, store also the negative status particles (holes)
    if self.jetscape:
        io_det_holes = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                           track_tree_name='tree_Particle', use_ev_id_ext=False,
                                            is_jetscape=self.jetscape, holes=True,
                                            event_plane_range=self.event_plane_range)
        df_fjparticles_det_holes = io_det_holes.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
        self.nEvents_det_holes = len(df_fjparticles_det_holes.index)
        self.nTracks_det_holes = len(io_det_holes.track_df.index)
        print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    # ------------------------------------------------------------------------

    # Use IO helper class to convert truth-level ROOT TTree into
    # a SeriesGroupBy object of fastjet particles per event
    io_truth = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                    track_tree_name='tree_Particle_gen', use_ev_id_ext=False,
                                    is_jetscape=self.jetscape, event_plane_range=self.event_plane_range, is_ENC=self.ENC_fastsim, is_det_level=False)
    df_fjparticles_truth = io_truth.load_data(m=self.m) # no dropping of tracks at truth level (important for the det-truth association because the index of the truth particle is used)
    self.nEvents_truth = len(df_fjparticles_truth.index)
    self.nTracks_truth = len(io_truth.track_df.index)
    print('--- {} seconds ---'.format(time.time() - self.start_time))

    print('Input truth Data Frame',df_fjparticles_truth)
    
    # If jetscape, store also the negative status particles (holes)
    if self.jetscape:
        io_truth_holes = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                              track_tree_name='tree_Particle_gen', use_ev_id_ext=False,
                                              is_jetscape=self.jetscape, holes=True,
                                              event_plane_range=self.event_plane_range)
        df_fjparticles_truth_holes = io_truth_holes.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
        self.nEvents_truth_holes = len(df_fjparticles_truth_holes.index)
        self.nTracks_truth_holes = len(io_truth_holes.track_df.index)
        print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    # ------------------------------------------------------------------------

    # Now merge the two SeriesGroupBy to create a groupby df with [ev_id, run_number, fj_1, fj_2]
    # (Need a structure such that we can iterate event-by-event through both fj_1, fj_2 simultaneously)
    # In the case of jetscape, we merge also the hole collections fj_3, fj_4
    print('Merge det-level and truth-level into a single dataframe grouped by event...')
    # print('debug df_fjparticles_det',df_fjparticles_det)
    # print('debug df_fjparticles_truth',df_fjparticles_truth)
    if self.jetscape:
        self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth, df_fjparticles_det_holes, df_fjparticles_truth_holes], axis=1)
        self.df_fjparticles.columns = ['fj_particles_det', 'fj_particles_truth', 'fj_particles_det_holes', 'fj_particles_truth_holes']
    elif self.ENC_fastsim:
        self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth], axis=1)
        self.df_fjparticles.columns = ['fj_particles_det', 'ParticleMCIndex', 'fj_particles_truth', 'ParticlePID']
        print('Merged output',self.df_fjparticles.columns)
        print(self.df_fjparticles)
    else:
        self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth], axis=1)
        self.df_fjparticles.columns = ['fj_particles_det', 'fj_particles_truth']
    print('--- {} seconds ---'.format(time.time() - self.start_time))

    # ------------------------------------------------------------------------
    
    # Set up the Pb-Pb embedding object
    if not self.is_pp and not self.thermal_model:
        self.process_io_emb = process_io_emb.ProcessIO_Emb(self.emb_file_list, track_tree_name='tree_Particle', m=self.m)
    
    # ------------------------------------------------------------------------

    self.initialize_output_objects()
    
    print(self)
    
    # Find jets and fill histograms
    print('Find jets...')
    self.analyze_events()
    
    # Plot histograms
    print('Save histograms...')
    process_base.ProcessBase.save_output_objects(self)
    
    print('--- {} seconds ---'.format(time.time() - self.start_time))

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_output_objects(self):

    # Initialize user-specific histograms
    self.initialize_user_output_objects()

    name = 'preprocessed_np_mc_jetpt'
    h = []
    setattr(self, name, h)
    
    self.hNevents = ROOT.TH1F('hNevents', 'hNevents', 2, -0.5, 1.5)
    self.hNevents.Fill(1, self.nEvents_det)
    
    self.hTrackEtaPhi = ROOT.TH2F('hTrackEtaPhi', 'hTrackEtaPhi', 200, -1., 1., 628, 0., 6.28)
    self.hTrackPt = ROOT.TH1F('hTrackPt', 'hTrackPt', 300, 0., 300.)
    
    if not self.is_pp:
      self.hRho =  ROOT.TH1F('hRho', 'hRho', 1000, 0., 1000.)
      self.hSigma = ROOT.TH1F('hSigma', 'hSigma', 100, 0., 100.)
      self.hDelta_pt = ROOT.TH1F('hDelta_pt', 'hDelta_pt', 200, -60, 120)
      
    if not self.skip_deltapt_RC_histograms:
      name = 'hN_MeanPt'
      h = ROOT.TH2F(name, name, 200, 0, 5000, 200, 0., 2.)
      setattr(self, name, h)

    # delta-eta-delta-phi distribution for truth and matched reco particles
    name = 'h2d_matched_part_deta_pt'
    eta_bins = linbins(-0.05, 0.05, 100)
    pt_bins = linbins(0, 10, 100)
    h = ROOT.TH2D(name, name, 100, eta_bins, 100, pt_bins)
    h.GetXaxis().SetTitle('#Delta#eta')
    h.GetYaxis().SetTitle('p_{T}')
    setattr(self, name, h)

    name = 'h2d_matched_part_dphi_pt'
    phi_bins = linbins(-0.02, 0.02, 100)
    pt_bins = linbins(0, 10, 100)
    h = ROOT.TH2D(name, name, 100, phi_bins, 100, pt_bins)
    h.GetXaxis().SetTitle('#Delta#phi')
    h.GetYaxis().SetTitle('p_{T}')
    setattr(self, name, h)

    # matching efficiency of reco tracks wrt truth tracks, as a function of pt
    name = 'h1d_truth_part_pt'
    pt_bins = linbins(0,10,200)
    h = ROOT.TH1D(name, name, 200, pt_bins)
    h.GetXaxis().SetTitle('p_{T}')
    setattr(self, name, h)

    name = 'h1d_det_part_pt'
    pt_bins = linbins(0,10,200)
    h = ROOT.TH1D(name, name, 200, pt_bins)
    h.GetXaxis().SetTitle('p_{T}')
    setattr(self, name, h)

    name = 'h1d_truth_matched_part_pt'
    pt_bins = linbins(0,10,200)
    h = ROOT.TH1D(name, name, 200, pt_bins)
    h.GetXaxis().SetTitle('p_{T}')
    setattr(self, name, h)

    name = 'h1d_det_matched_part_pt'
    pt_bins = linbins(0,10,200)
    h = ROOT.TH1D(name, name, 200, pt_bins)
    h.GetXaxis().SetTitle('p_{T}')
    setattr(self, name, h)

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
  # Main function to loop through and analyze events
  #---------------------------------------------------------------
  def analyze_events(self):
    
    fj.ClusterSequence.print_banner()
    print()
    
    self.event_number = 0
    
    # Then can use list comprehension to iterate over the groupby and do jet-finding
    # simultaneously for fj_1 and fj_2 per event, so that I can match jets -- and fill histograms
    if self.jetscape:
        result = [self.analyze_event(fj_particles_det, fj_particles_truth, fj_particles_det_holes, fj_particles_truth_holes) for fj_particles_det, fj_particles_truth, fj_particles_det_holes, fj_particles_truth_holes in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth'], self.df_fjparticles['fj_particles_det_holes'], self.df_fjparticles['fj_particles_truth_holes'])]
    elif self.ENC_fastsim:
        result = [self.analyze_event(fj_particles_det=fj_particles_det, fj_particles_truth=fj_particles_truth, particles_mcid_det=particles_mcid_det, particles_pid_truth=particles_pid_truth) for fj_particles_det, fj_particles_truth, particles_mcid_det, particles_pid_truth in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth'], self.df_fjparticles['ParticleMCIndex'], self.df_fjparticles['ParticlePID'])]
    else:
        result = [self.analyze_event(fj_particles_det, fj_particles_truth) for fj_particles_det, fj_particles_truth in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth'])]
    
    if self.debug_level > 0:
      for attr in dir(self):
        obj = getattr(self, attr)
        print('size of {}: {}'.format(attr, sys.getsizeof(obj)))
    
    print('Save thn...')
    process_base.ProcessBase.save_thn_th3_objects(self)
    
  #---------------------------------------------------------------
  # Fill track histograms.
  #---------------------------------------------------------------
  def fill_track_histograms(self, fj_particles_det):

    # Check that the entries exist appropriately
    # (need to check how this can happen -- but it is only a tiny fraction of events)
    if type(fj_particles_det) != fj.vectorPJ:
      return
    
    for track in fj_particles_det:
      self.hTrackEtaPhi.Fill(track.eta(), track.phi())
      self.hTrackPt.Fill(track.pt())
      
  #---------------------------------------------------------------
  # Analyze jets of a given event.
  # fj_particles is the list of fastjet pseudojets for a single fixed event.
  #---------------------------------------------------------------
  def analyze_event(self, fj_particles_det, fj_particles_truth, fj_particles_det_holes=None, fj_particles_truth_holes=None, particles_mcid_det=None, particles_pid_truth=None):
    
    self.event_number += 1
    if self.event_number > self.event_number_max:
      return
    if self.debug_level > 1:
      print('-------------------------------------------------')
      print('event {}'.format(self.event_number))
        
    if self.event_number % 1000 == 0: print("analyzing event : " + str(self.event_number))

    # various checks
    if self.jetscape:
      if type(fj_particles_det_holes) != fj.vectorPJ or type(fj_particles_truth_holes) != fj.vectorPJ:
        print('fj_particles_holes type mismatch -- skipping event')
        return

    #handle case with no truth particles
    if type(fj_particles_truth) is float:
        print("EVENT WITH NO TRUTH PARTICLES!!!")
        return
    
    #handle case with no det particles
    if type(fj_particles_det) is float:
        print("EVENT WITH NO DET PARTICLES!!!")
        fj_particles_det = []

    if len(fj_particles_truth) > 1:
      if np.abs(fj_particles_truth[0].pt() - fj_particles_truth[1].pt()) <  1e-10:
        print('WARNING: Duplicate particles may be present')
        print([p.user_index() for p in fj_particles_truth])
        print([p.pt() for p in fj_particles_truth])

    ##### CHARGE PARTICLE AND PARTICLE PT > 0.15 CUT
        # ALSO RESET ALL USER INDICIES to 0
    fj_particles_det_pass = fj.vectorPJ()
    for part in fj_particles_det:
      if part.perp() > 0.15:
        if not self.ENC_fastsim or (self.ENC_fastsim and part.python_info().charge != 0):
          part.set_user_index(0)
          fj_particles_det_pass.push_back(part)
    fj_particles_det = fj_particles_det_pass

    fj_particles_truth_pass = fj.vectorPJ()
    for part in fj_particles_truth:
      if part.perp() > 0.15:
        if not self.ENC_fastsim or (self.ENC_fastsim and part.python_info().charge != 0):
          part.set_user_index(0)
          fj_particles_truth_pass.push_back(part)
    fj_particles_truth = fj_particles_truth_pass


    for det_part in fj_particles_det:
        hname = 'h1d_det_part_pt'
        getattr(self, hname).Fill(det_part.perp())


    ############################# TRACK MATCHING ################################
    # set all indicies to dummy index
    dummy_index = -1
    for i in range(len(fj_particles_truth)):
      fj_particles_truth[i].set_user_index(dummy_index)
    for i in range(len(fj_particles_det)):
      fj_particles_det[i].set_user_index(dummy_index)
    
    # perform matching, give matches the same user_index
    index = 1
    det_used = []
    # note: CANNOT loop like this: <for truth_part in fj_particles_truth:>
    for itruth in range(len(fj_particles_truth)):
      truth_part = fj_particles_truth[itruth]
    
      truth_part.set_user_index(index)
        
      candidates = []
      candidates_R = []
        
      for idet in range(len(fj_particles_det)):
        det_part = fj_particles_det[idet]
        
        delta_R = self.calculate_distance(truth_part, det_part)
        if delta_R < 0.05 and abs((det_part.perp() - truth_part.perp()) / truth_part.perp()) < 0.1 \
                and det_part not in det_used:
            candidates.append(det_part)
            candidates_R.append(delta_R)

      # if match found
      if len(candidates) > 0:
        det_match = candidates[np.argmin(candidates_R)]
        det_match.set_user_index(index)
        det_used.append(det_match)

        deta = det_match.eta() - truth_part.eta()
        dphi = det_match.phi() - truth_part.phi()

        hname = 'h2d_matched_part_deta_pt'
        getattr(self, hname).Fill(deta, truth_part.perp())

        hname = 'h2d_matched_part_dphi_pt'
        getattr(self, hname).Fill(dphi, truth_part.perp())

        hname = 'h1d_truth_matched_part_pt'
        getattr(self, hname).Fill(truth_part.perp())
        
        hname = 'h1d_det_matched_part_pt'
        getattr(self, hname).Fill(det_match.perp())


      hname = 'h1d_truth_part_pt'
      getattr(self, hname).Fill(truth_part.perp())

      index += 1

    # handle unmatched particles, give them all different user_index s
    for i in range(len(fj_particles_truth)):
      part = fj_particles_truth[i]
      if part.user_index() == dummy_index:
        part.set_user_index(index)
        index += 1

    unmatched_det = 0
    for i in range(len(fj_particles_det)):
      part = fj_particles_det[i]
      if part.user_index() == dummy_index:
        part.set_user_index(index)
        index += 1
        unmatched_det += 1
    # print("det matches: {} / {} = {}".format(unmatched_det, len(fj_particles_det), unmatched_det / len(fj_particles_det)))

    ######################### CONSTRUCT ENBEDDED EVENT ##########################
    # positive index = signal particle
    # negative index = bkgd particle
    index = -1
    if not self.is_pp:
      # If Pb-Pb, construct embedded event (do this once, for all jetR)
      # For thermal model, look at previous commits

      # Main case: Get Pb-Pb event and embed it into the det-level particle list
      fj_particles_hybrid = self.process_io_emb.load_event()

      # give each particle new unique user_index
      for part in fj_particles_hybrid:
        part.set_user_index(index)
        index -= 1
          
      # Form the combined det-level event
      _ = [fj_particles_hybrid.push_back(p) for p in fj_particles_det]

    """ For debugging """
    """
    det_pt_output = "DETECTOR PARTICLES pt: "
    det_id_output = "DETECTOR PARTICLES id: "
    for d_part in fj_particles_det:
        det_pt_output += str(d_part.perp()) + " "
        det_id_output += str(d_part.user_index()) + " "
    print(det_pt_output)
    print(det_id_output)
    
    truth_pt_output = "TRUTH PARTICLES pt: "
    truth_id_output = "TRUTH PARTICLES id: "
    for d_part in fj_particles_truth:
        truth_pt_output += str(d_part.perp()) + " "
        truth_id_output += str(d_part.user_index()) + " "
    print(truth_pt_output)
    print(truth_id_output)
    """    

    ############################# JET RECO ################################
    jetR = self.jetR

    # Set jet definition and a jet selector
    jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
    jet_selector_det = fj.SelectorPtMin(self.jetpt_min_det) & fj.SelectorAbsRapMax(0.9 - jetR)
    jet_selector_truth = fj.SelectorPtMin(self.jetpt_min_truth) & fj.SelectorAbsRapMax(0.9)

    if self.debug_level > 2:
      print('')
      print('jet definition is:', jet_def)
      print('jet selector for det-level is:', jet_selector_det)
      print('jet selector for truth-level matches is:', jet_selector_truth)

    # cluster truth jets
    cs_truth = fj.ClusterSequence(fj_particles_truth, jet_def)
    truth_jets = fj.sorted_by_pt(jet_selector_truth(cs_truth.inclusive_jets()))

    # KD: no cuts should be applied on the truth jets! only det and hybrid jets
    # this includes no leading track or jet area cuts

    # cluster detector/hybrid jets
    if self.is_pp:
      
      # cs_det = fj.ClusterSequenceArea(fj_particles_det, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
      cs_det = fj.ClusterSequence(fj_particles_det, jet_def)
      det_jets = fj.sorted_by_pt(jet_selector_det(cs_det.inclusive_jets()))

      # KD: EEC and jet-trk preprocessed output
      self.analyze_jets(det_jets, truth_jets, jetR)
      
    else:

      # cluster embedded jets and jet pT correction
      
      # rho calculation
      bge = fj.GridMedianBackgroundEstimator(0.9, 0.4) # max eta, grid size
      bge.set_particles(fj_particles_hybrid)
      rho = bge.rho()
      sigma = bge.sigma()

      getattr(self, "hRho").Fill(rho)
      getattr(self, "hSigma").Fill(sigma)
      
      cs_hybrid = fj.ClusterSequenceArea(fj_particles_hybrid, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
      hybrid_jets = fj.sorted_by_pt(jet_selector_det(cs_hybrid.inclusive_jets()))

      self.analyze_jets(hybrid_jets, truth_jets, jetR, rho=rho)


  def analyze_jets(self, jets_det, jets_truth, jetR, rho = 0):
    
    raise NotImplementedError('You must implement analyze_jets()!')

  #---------------------------------------------------------------
  # This function is called once for each jetR
  # You must implement this
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
      
    raise NotImplementedError('You must implement initialize_user_output_objects()!')
