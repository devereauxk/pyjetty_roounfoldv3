#!/usr/bin/env python3

"""
Base class to read a ROOT TTree of track information
and do jet-finding, and save basic histograms.
  
To use this class, the following should be done:

  - Implement a user analysis class inheriting from this one, such as in user/james/process_data_XX.py
    You should implement the following functions:
      - initialize_user_output_objects()
      - fill_jet_histograms()
    
  - The histogram of the data should be named h_[obs]_JetPt_R[R]_[subobs]_[grooming setting]
    The grooming part is optional, and should be labeled e.g. zcut01_B0 — from CommonUtils::grooming_label({'sd':[zcut, beta]})
    For example: h_subjet_z_JetPt_R0.4_0.1
    For example: h_subjet_z_JetPt_R0.4_0.1_zcut01_B0

  - You also should modify observable-specific functions at the top of common_utils.py
  
Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import time

# Data analysis and plotting
import numpy as np
import ROOT
import yaml
import pandas as pd
import math

# Fastjet via python (from external library heppy)
import fastjet as fj

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_base

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
class ProcessDataBase(process_base.ProcessBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessDataBase, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

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

    if 'use_ev_id' in config:
      self.use_ev_id = config['use_ev_id']
    else:
      self.use_ev_id = True

    self.is_pp = config['is_pp']

    if 'do_perpcone' in config:
      self.do_perpcone = config['do_perpcone']
    else:
      self.do_perpcone = False

    if 'do_rho_subtraction' in config:
      self.do_rho_subtraction = config['do_rho_subtraction']
    else:
      self.do_rho_subtraction = False

    if 'jetpt_min_det' in config:
      self.jetpt_min_det = config['jetpt_min_det']
    else:
      self.jetpt_min_det = 10

    if 'jetpt_min_det_subtracted' in config:
      self.jetpt_min_det_subtracted = config['jetpt_min_det_subtracted']
    else:
      self.jetpt_min_det_subtracted = 10

    if 'jetR' in config:
      self.jetR = config['jetR']
    else:
      self.jetR = 0.4

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
  def process_data(self):
    
    self.start_time = time.time()

    # Use IO helper class to convert ROOT TTree into a SeriesGroupBy object of fastjet particles per event
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    io = process_io.ProcessIO(input_file=self.input_file, track_tree_name='tree_Particle',
                              is_pp=self.is_pp, use_ev_id_ext=self.use_ev_id)
    self.df_fjparticles = io.load_data(m=self.m)
    self.nEvents = len(self.df_fjparticles.index)
    self.nTracks = len(io.track_df.index)
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    # Initialize histograms
    self.initialize_output_objects()
    
    print(self)

    # Find jets and fill histograms
    print('Analyze events...')
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
    
    # Initialize base histograms
    self.hNevents = ROOT.TH1F('hNevents', 'hNevents', 2, -0.5, 1.5)
    if self.event_number_max < self.nEvents:
      self.hNevents.Fill(1, self.event_number_max)
    else:
      self.hNevents.Fill(1, self.nEvents)
    
    self.hTrackEtaPhi = ROOT.TH2F('hTrackEtaPhi', 'hTrackEtaPhi', 200, -1., 1., 628, 0., 6.28)
    self.hTrackPt = ROOT.TH1F('hTrackPt', 'hTrackPt', 300, 0., 300.)
    
    if not self.is_pp:
      self.hRho = ROOT.TH1F('hRho', 'hRho', 100, 0., 300.)
      self.hSigma = ROOT.TH1F('hSigma', 'hSigma', 100, 0., 100.)
      self.hDelta_pt = ROOT.TH1F('hDelta_pt', 'hDelta_pt', 200, -60, 120)

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
    
    # Fill track histograms
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    print('Fill track histograms')
    [[self.fillTrackHistograms(track) for track in fj_particles] for fj_particles in self.df_fjparticles]
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    print('Find jets...')
    fj.ClusterSequence.print_banner()
    print()
    self.event_number = 0
  
    # Use list comprehension to do jet-finding and fill histograms
    _ = [self.analyze_event(fj_particles) for fj_particles in self.df_fjparticles]
    
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    print('Save thn...')
    process_base.ProcessBase.save_thn_th3_objects(self)
  
  #---------------------------------------------------------------
  # Fill track histograms.
  #---------------------------------------------------------------
  def fillTrackHistograms(self, track):
    
    self.hTrackEtaPhi.Fill(track.eta(), track.phi())
    self.hTrackPt.Fill(track.pt())
  
  #---------------------------------------------------------------
  # Analyze jets of a given event.
  # fj_particles is the list of fastjet pseudojets for a single fixed event.
  #---------------------------------------------------------------
  def analyze_event(self, fj_particles):
    
    self.event_number += 1
    if self.event_number > self.event_number_max:
      return
    if self.debug_level > 1:
      print('-------------------------------------------------')
      print('event {}'.format(self.event_number))

    if self.event_number % 1000 == 0: print("analyzing event : " + str(self.event_number))

    #handle case with no truth particles
    if type(fj_particles) is float:
        print("EVENT WITH NO PARTICLES!!!")
        return
    
    # various checks
    if len(fj_particles) > 1:
      if np.abs(fj_particles[0].pt() - fj_particles[1].pt()) <  1e-10:
        print('WARNING: Duplicate particles may be present')
        print([p.user_index() for p in fj_particles])
        print([p.pt() for p in fj_particles])
    
    ##### PARTICLE PT CUT > 0.15
    fj_particles_pass = fj.vectorPJ()
    for part in fj_particles:
      if part.perp() > 0.15:
        fj_particles_pass.push_back(part)
    fj_particles = fj_particles_pass
  

    ############################# JET RECO ################################
    jetR = self.jetR

    # Set jet definition and a jet selector
    jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
    jet_selector = fj.SelectorPtMin(self.jetpt_min_det) & fj.SelectorAbsRapMax(0.9 - jetR)

    if self.debug_level > 2:
        print('jet definition is:', jet_def)
        print('jet selector is:', jet_selector,'\n')

    # Analyze
    if self.is_pp:

      #cs = fj.ClusterSequenceArea(fj_particles, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
      cs = fj.ClusterSequence(fj_particles, jet_def)
      jets_selected = fj.sorted_by_pt(jet_selector(cs.inclusive_jets()))

      # KD: EEC and jet-trk preprocessed output
      self.analyze_jets(jets_selected, jetR, fj_particles)

    else:
      
      # performs rho subtraction automatically

      # embeding random 60GeV particle with unique user_index=-3
      # TODO should I remove this when running fr?
      random_Y = np.random.uniform(-0.9, 0.9)
      random_phi = np.random.uniform(0, 2*np.pi)
      probe = fj.PseudoJet()
      probe.reset_PtYPhiM(60, random_Y, random_phi, 1)
      probe.set_user_index(-3)
      fj_particles.push_back(probe)

      # rho calculation
      bge = fj.GridMedianBackgroundEstimator(0.9, 0.4) # max eta, grid size
      bge.set_particles(fj_particles)
      rho = bge.rho()
      sigma = bge.sigma()

      # print(" RHO THIS EVENT : {}, +/- {}".format(rho, sigma)) 

      getattr(self, "hRho").Fill(rho)
      getattr(self, "hSigma").Fill(sigma)

      cs = fj.ClusterSequenceArea(fj_particles, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
      jets_selected = fj.sorted_by_pt(jet_selector(cs.inclusive_jets()))

      self.analyze_jets(jets_selected, jetR, fj_particles, rho=rho)


  #---------------------------------------------------------------
  # This function is called once
  # You must implement this
  #---------------------------------------------------------------
  def analyze_jets(self, jets, jetR, fj_particles, rho = 0):

    raise NotImplementedError('You must implement analyze_jets()!')

  #---------------------------------------------------------------
  # This function is called once
  # You must implement this
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
  
    raise NotImplementedError('You must implement initialize_user_output_objects()!')
