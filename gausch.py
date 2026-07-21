#!/usr/bin/env python3
# ==============================================================
# Copyright(c) 2026-, Pei-Kang Tsou (tpkcpk@gmail.com)
# Creation Time : 20260721 10:00:00
# ==============================================================
from sys import argv
from os import listdir

import torch, copy
import schnetpack as spk
import numpy as np
from ase.io import read
from ase import Atoms


def preprocess():
    pref = {}
# ======= Default pref defined in here =======
    pref["nnp_path"] = "./"
    pref["stats_pt"] = 'property_stats.pt'
    pref["model"] = 'best'
    
    pref["set_value"] = 128
    pref["n_gaussians"] = 30
    pref["n_interactions"] = 4
    pref["cutoff"] = 15.0
    pref["device"] = "cpu"
    pref["B2A"] = 0.52917725
    pref["mpath"] = '/dev/shm/schnet-g16/'
    
    pref["atomref"] = 0
    pref["atomref_dict"] = {1: -0.4981341, 6: -37.7769172, 16:-398.0369383,
                         7:-54.4762936, 8:-74.9650351, 11:-162.2395169}
                         
    pref['atoms'] = { 1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 11:'Na', 3:'Li', 16:'S', 19:'K', 17:'Cl'}
    pref["nnp_path"] = pref["nnp_path"] + '/' if pref["nnp_path"][-1] != '/' else pref["nnp_path"]
    return pref 
    
    
# ======= Function starts =====================
def gauinput(inp):
    f = open(inp, 'r')
    fr = f.readlines() 
    f.close()
    info = ' '.join(fr[0].split()).split(' ')
    nums=[]
    poss=[]
    for j in range(1,int(info[0])+1):
      l = ' '.join(fr[j].split()).split(' ')
      nums.append(l[0])
      poss.append((float(l[1])*pref["B2A"], float(l[2])*pref["B2A"], float(l[3])*pref["B2A"]))
    return Atoms(numbers=nums,positions=poss), info[0], info[1]
    
   
def schnet_model():
  if pref["atomref"]:
    atomrefs = np.zeros((100, 1))
    for z, en in pref["atomref_dict"].items():
        atomrefs[z] = en
    atomrefs = {'energy': atomrefs}
    md_aref = atomrefs["energy"]
  
  if pref["model"] == 'best':
      print (f"Process {pref['nnp_path']}best_model")
      the_model = torch.load(pref['nnp_path']+'best_model', map_location=torch.device(pref['device']),weights_only=False)
      
  else:
      prop_stats = torch.load(pref['nnp_path']+pref["stats_pt"])
      means, stddevs = prop_stats["means"], prop_stats["stddevs"]
      ckpt_list = [i for i in listdir(pref['nnp_path']) if i[-8:] == '.pth.tar']
      ckpt_list = [i for i in ckpt_list if i[:3] != 'pm_']
      ckpt_path = pref['nnp_path'] + ckpt_list[-1]
      print(f"Process {ckpt_path}")

      in_module = spk.representation.SchNet(n_atom_basis=pref["set_value"],n_filters=pref["set_value"],n_gaussians=pref["n_gaussians"],charged_systems=True,n_interactions=pref["n_interactions"],cutoff=pref["cutoff"],cutoff_network=spk.nn.cutoff.CosineCutoff)        
      out_module = spk.atomistic.Atomwise(n_in=pref["set_value"], property="energy",mean=means["energy"],stddev=stddevs["energy"],derivative="force",atomref=md_aref,negative_dr=True)

      the_model = spk.AtomisticModel(representation=in_module, output_modules=out_module)
      the_model.load_state_dict(torch.load(ckpt_path, map_location=pref["device"])["model"])
      the_model.to(pref["device"])
      the_model.eval()        
  calculator = spk.interfaces.SpkCalculator(model=the_model,device=pref["device"],energy='energy',forces='force')                                      
  return calculator
    
def predictor(atoms, nat, fqm, output):
  atoms.calc = copy.deepcopy(schnet_model())
  energy = atoms.get_potential_energy()
  force = atoms.get_forces()
  
  wout = fval(energy)+fval(0.0)+fval(0.0)+fval(0.0)+'\n'  ##energy[0]
  for fi, fo in enumerate(force):
    for f in fo:
      wout = wout + fval(-1 * f * pref["B2A"])
    wout = wout + '\n'

  if fqm == '2':
    from ase.phonons import Phonons
    import random
    import os
    tmp = str(100000+int(random.random() * 100000))
    tmp_path = pref["mpath"] + tmp
    os.system(f"mkdir -p {tmp_path}")
    ph = Phonons(atoms, calc=atoms.calc, name=f"{tmp_path}")
    
    ph.run()
    ph.read(acoustic=True) 
    forceC = ph.get_force_constant()
    forceC = forceC[0]

    for i in range(2):  
      wout = wout + fval(0.0)+fval(0.0)+fval(0.0)+'\n'
    for i in range(3*int(nat)):  
      wout = wout + fval(0.0)+fval(0.0)+fval(0.0)+'\n'
    num=0   
    for i in range(len(forceC)):
      for j in range(i+1):
        wout = wout + fval(forceC[i][j] * 0.28 )
        num+=1
        if num == 3:
          wout = wout + '\n'
          num=0  
    os.system(f"rm -rf {tmp_path}")
    
  fw = open(output, 'w')
  fw.writelines(wout)         
  fw.close()
    
def fval(val):
  return '{:20.12E}'.format(val)
    
if __name__ == "__main__":
  pref = preprocess()
  #data = {'input':argv[2],'output':argv[3],'msg':argv[4]}
  atoms, atomnum, fqm = gauinput(argv[2])
  predictor(atoms, atomnum, fqm, argv[3])


