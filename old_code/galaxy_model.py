#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  a template for a simple galaxy formation model a la Krumholz & Dekel (2012); 
#                    see also Feldmann 2013
#  used as part of the 2016 A304 "Galaxies" class
#
#   Andrey Kravtsov, May 2016
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from colossus.cosmology import cosmology
from scipy.interpolate import UnivariateSpline

def fg_in(Mh):
    return 1.0

def R_loss(dt):
    """
    fraction of mass formed in stars that is returned back to the ISM
    """
    return 0.46

class model_galaxy(object):

    def __init__(self,  t = None, Mh = None, Mg = None, Ms = None, MZ = None, Z_IGM = 1.e-4, sfrmodel = None, cosmo = None, verbose = False):

        self.Zsun = 0.02

        if cosmo is not None: 
            self.cosmo = cosmo
            self.fbuni = cosmo.Ob0/cosmo.Om0
        else:
            errmsg = 'to initialize gal object it is mandatory to supply the collossus cosmo(logy) object!'
            raise Exception(errmsg)
            return
            
        if Mh is not None: 
            self.Mh = Mh
        else:
            errmsg = 'to initialize gal object it is mandatory to supply Mh!'
            raise Exception(errmsg)
            return
            
        if t is not None: 
            self.t = t # in Gyrs
            self.z = self.cosmo.age(t, inverse=True)        
            self.gr = self.cosmo.growthFactor(self.z)
            self.dDdz = self.cosmo.growthFactor(self.z,derivative=1)
            self.thubble = self.cosmo.hubbleTime(self.z)
            self.dDdt = self.cosmo.growthFactor(self.z, derivative=1) * self.cosmo.age(t, derivative=1, inverse=True)

        else:
            errmsg = 'to initialize gal object it is mandatory to supply t!'
            raise Exception(errmsg)
            return
            
        # metallicity yield of a stellar population - this is a constant derived from SN and AGB simulations
        self.yZ = 0.069; 
        # assumed metallicity of freshly accreting gas (i.e. metallicity of intergalactic medium)
        self.Z_IGM = Z_IGM; 
        
        if Ms is not None:
            self.Ms = Ms
        else: 
            self.Ms = 0.0
        if Mg is not None:
            self.Mg = Mg
        else: 
            self.Mg = self.fbuni*Mh
            
        if MZ is not None:
            self.MZ = MZ
        else: 
            self.MZ = self.Z_IGM*self.Mg
        if MZ is not None and Mg is not None:
            # model for molecular hydrogen content is to be implemented here
            self.MH2 = 0.0
        else:
            self.MH2 = 0.0
        
        # only one model based on total gas density for starters, a better model is to be implemented
        #self.sfr_models = {'gaslinear': self.SFRgaslinear, 'H2linear': self.SFRlinear}
        self.sfr_models = {'gaslinear': self.SFRgaslinear}
    
        if sfrmodel is not None: 
            try: 
                self.sfr_models[sfrmodel]
            except KeyError:
                print "unrecognized sfrmodel in model_galaxy.__init__:", sfrmodel
                print "available models:", self.sfr_models
                return
            self.sfrmodel = sfrmodel
        else:
            errmsg = 'to initialize gal object it is mandatory to supply sfrmodel!'
            raise Exception(errmsg)
            return
            
        if verbose is not None:
            self.verbose = verbose
        else:
            errmsg = 'verbose parameter is not initialized'
            raise Exception(errmsg)
            return

        self.epsout = self.eps_out(); self.Mgin = self.Mg_in(t); 
        self.Msin = self.Ms_in(t)
        self.sfr = self.SFR(t)
        self.Rloss = R_loss(0.); self.Rloss1 = 1.0-self.Rloss

        return
        
    
    def dMhdt(self, Mcurrent, t):
        """
        halo mass accretion rate using approximation of eqs 3-4 of Krumholz & Dekel 2012
        output: total mass accretion rate in Msun/h /Gyr
        """
        self.Mh = Mcurrent
        # this equation is eq. 1 from Feldmann 2013
        dummy = 1.06e12*(Mcurrent/1.e12)**1.14 *self.dDdt/(self.gr*self.gr)

        # approximation in Krumholz & Dekel (2012) for testing 
        #dummy = 5.02e10*(Mcurrent/1.e12)**1.14*(1.+self.z+0.093/(1.+self.z)**1.22)**2.5    

        return dummy
                
    def eps_in(self, t):
        """
        fraction of universal baryon fraction that makes it into galaxy 
        along with da
        """
        epsin = 1.0
        return epsin

    def Mg_in(self, t):
        dummy = self.fbuni*self.eps_in(t)*fg_in(self.Mh)*self.dMhdt(self.Mh,t)
        return dummy
    
    def Ms_in(self, t):
        dummy = self.fbuni*(1.0-fg_in(self.Mh))*self.dMhdt(self.Mh,t)
        return dummy

    def tau_sf(self):
        """
        gas consumption time in Gyrs 
        """
        return 1.

    def SFRgaslinear(self, t):
        return self.Mg/self.tau_sf()
         
    def SFR(self, t):
        """
        master routine for SFR - 
        eventually can realize more star formation models
        """  
        return self.sfr_models[self.sfrmodel](t)
        
    def dMsdt(self, Mcurrent, t):
        dummy = self.Msin + self.Rloss1*self.sfr
        return dummy

    def eps_out(self):
        return 0.0
        
    def dMgdt(self, Mcurrent, t):
        dummy = self.Mgin - (self.Rloss1 + self.epsout)*self.sfr
        return dummy

    def zeta(self):
        """
        output: fraction of newly produced metals removed by SNe in outflows
        """
        return 0.0

    def dMZdt(self, Mcurrent, t):
        dummy = self.Z_IGM*self.Mgin + (self.yZ*self.Rloss1*(1.-self.zeta()) - (self.Rloss1+self.epsout)*self.MZ/(self.Mg))*self.sfr
        return dummy
        
    def evolve(self, Mcurrent, t):
        # first set auxiliary quantities and current masses
        self.z = self.cosmo.age(t, inverse=True)        
        self.gr = self.cosmo.growthFactor(self.z)
        self.dDdz = self.cosmo.growthFactor(self.z,derivative=1)
        self.thubble = self.cosmo.hubbleTime(self.z)
        self.dDdt = self.cosmo.growthFactor(self.z, derivative=1) * self.cosmo.age(t, derivative=1, inverse=True)

        self.Mh = Mcurrent[0]; self.Mg = Mcurrent[1]; 
        self.Ms = Mcurrent[2]; self.MZ = Mcurrent[3]
        self.epsout = self.eps_out(); self.Mgin = self.Mg_in(t); 
        self.Msin = self.Ms_in(t)
        self.Rloss = R_loss(0.); self.Rloss1 = 1.0-self.Rloss
        self.sfr = self.SFR(t)
        
        # calculate rates for halo mass, gas mass, stellar mass, and mass of metals
        dMhdtd = self.dMhdt(Mcurrent[0], t)
        dMgdtd = self.dMgdt(Mcurrent[1], t)
        dMsdtd = self.dMsdt(Mcurrent[2], t)
        dMZdtd = self.dMZdt(Mcurrent[3], t)
        if self.verbose:
            print "evolution: t=%2.3f Mh=%.2e, Mg=%.2e, Ms=%.2e, Z/Zsun=%2.2f,SFR=%4.1f"%(t,self.Mh,self.Mg,self.Ms,self.MZ/self.Mg/0.02,self.SFR(t)*1.e-9)

        return [dMhdtd, dMgdtd, dMsdtd, dMZdtd]


