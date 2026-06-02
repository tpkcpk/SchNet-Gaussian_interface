#!/usr/bin/env python3
# ==============================================================
# Copyright(c) 2026-, Pei-Kang Tsou (tpkcpk@gmail.com)
# Creation Time : 20260602 10:00:00
# ==============================================================
from sys import argv
from os import path, remove, listdir

from ase.phonons import Phonons
import torch
import schnetpack as spk
import numpy as np
from ase.io import read

def preprocess():
    pref = {}
    pref.update(func_info())
# ======= Default pref defined in here =======
    pref["nnp_path"] = "/home/pktsou/lustre/nn_schnet/"
    pref["stats_pt"] = 'property_stats.pt'
    pref["model"] = 'best'
    
    pref["set_value"] = 128
    pref["n_gaussians"] = 30
    pref["n_interactions"] = 4
    pref["cutoff"] = 15.0
    pref["device"] = "cpu"
    pref["B2A"] = 0.52917725
    
    
    pref["nnp_path"] = pref["nnp_path"] + '/' if nnp_path[-1] != '/' else pref["nnp_path"]
    return pref 
    
    
# ======= Function starts =====================
def gauinput(nat, inp):
    atom_d={'1':'H','6':'C','8':'O','11':'Na','7':'N'}
    fr = open(inp+'.txt', 'r')
    i = fr.read().split(' ')
    fr.close()
    fw = open(inp+'.xyz', 'w')
    fw.writelines(f"{nat}\ngausch\n")
    
    for j in range(int(nat)):
      fw.writelines(atom_d.get(i[4*j]))
      for k in range(1,4):
      	fw.writelines('\t'+'{:f}'.format(float(i[4*j+k])*0.5292))
      fw.writelines('\n')	
    fw.close()
    property_ls = [{"energy": np.array([0], dtype="float32"),
               "force": np.array([0], dtype="float32")}]
    mols = read(inp+'.xyz', index=":")
    dataset = spk.AtomsData(inp+'.db', available_properties=pref["properties"])
    dataset.add_systems(mols,property_ls)
    #return mols[0]
   
def predictor(pref, nat, fqm, output,inp):
    print ('nnp_path=', pref['nnp_path'])
    
    if pref["atomref"]:
      atomrefs = np.zeros((100, 1))
      for z, en in pref["atomref_dict"].items():
          atomrefs[z] = en
      atomrefs = {'energy': atomrefs}
      md_aref = atomrefs["energy"]
    
    if pref["model"] == 'best':
        print ('Process best_model')
        the_model = torch.load(pref['nnp_path']+'best_model', map_location=torch.device(pref['device']), weights_only=False)
        
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
                                      
                                      
                                      
                                      
                                      
    db = spk.AtomsData(inp+'.db')  
    atoms, properties = db.get_properties(idx=0)
    
    atoms.set_calculator(calculator)
    energy = atoms.get_potential_energy()[0]
    force = atoms.get_forces()
    fw = open(output, 'w')
    fw.writelines(fval(energy)+fval(0.0)+fval(0.0)+fval(0.0)+'\n')
    for fo in force:
      for f in fo:
        fw.writelines(fval(-1 * f * 0.5292))
      fw.writelines('\n')
      
    if fqm == '2':
      ph = Phonons(atoms, calc=calculator, name='/dev/shm/'+inputpath)
      ph.run()
      ph.read(acoustic=True) 
      forceC = ph.get_force_constant()
      forceC = forceC[0]
    
      for i in range(2):  
        fw.writelines(fval(0.0)+fval(0.0)+fval(0.0)+'\n')
      for i in range(3*int(nat)):  
        fw.writelines(fval(0.0)+fval(0.0)+fval(0.0)+'\n')
      num=0   
      for i in range(len(forceC)):
        for j in range(i+1):
          fw.writelines(fval(forceC[i][j] * 0.28))
          #fw.writelines(fval(forceC[i][j]))
          num+=1
          if num == 3:
            fw.writelines('\n')
            num=0
    fw.close()
    
def fval(val):
    return '{:20.12E}'.format(val) #.replace('E','D')
    
if __name__ == "__main__":
  pref = preprocess()
  #data = {{'filename':argv[1],'input':argv[3],'output':argv[4]}}
  
  atoms = argv[1]
  freq_mode = argv[2]
  output = argv[3]
  inputpath = argv[4]
  
  atoms, atomnum, fqm = gauinput(params['input'])
  predictor(nnp, atoms, atomnum, fqm, params['output'], params['filename'])
  

  
  
  gauinput(n_atom, '/dev/shm/'+inputpath)
  predictor(pref, n_atom, freq_mode, output, '/dev/shm/'+inputpath)
  if freq_mode == '2':
     for f in listdir('./'):
       if f.startswith(inputpath):
           remove(f)

