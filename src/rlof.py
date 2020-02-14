import numpy as np
from astropy.table import Table
from scipy.integrate import solve_ivp
from Constants import Constants


class RLOF:
    """
    CLASS RLOF:
    Initialize a binary system with set properties, then integrate its evolution during Roche lobe overflow.
    
    Example:
    
    >>> r = RLOF(Md0=2.e33,
         Ma0=1.e33,
         Rd0=7e12,
         Ra0=0.0,
         Ggrav=6.674e-8,
         a0_mode='Roche_limit_fraction',
         f_roche_limit=0.9,
         gamma_adiabatic=5./3.,
         gamma_structure=5./3.,
         fcorot0=1.0)

    >>> r.integrate(2e7,Ntimes=1001)
    
    """
    def __init__(self,
                 Md0=2.e33,
                 Ma0=1.e33,
                 Rd0=7e10,
                 Ra0=7e8,
                 Ggrav=6.674e-8,
                 a0_mode = 'Roche_limit_fraction',
                 a0 = 1.e11,
                 P0=86400.0,
                 f_roche_limit=0.99,
                 Rdfunc=None,
                 fcorot0=1.0,
                 gamma_adiabatic=5./3.,
                 gamma_structure=5./3.,
                 amin_integrate=None,
                 Mdtot=None,
                 verbose=True):
        
        """
        Parameters:
        
        Md0=2.e33,              # initial donor mass
        
        Ma0=1.e33,              # initial accretor mass
        
        Rd0=7e10,               # initial donor radius
        
        Ra0=0.0,                # initial accretor radius (must be >0 if acc_mode='eddington' in integrate)
        
        Ggrav=6.674e-8,         # gravitational constant (sets unit system)
        
        a0_mode = 'Roche_limit_fraction',   # how to initialize the separation, options: manual, Roche_limit_fraction, period
        
        a0 = 1.e11,             # initial separation, if set with a0_mode='manual'
        
        P0 = 86400.0            # initial target orbital period, if a0_mode='period'
        
        f_roche_limit=0.99,     # initial fraction of Roche limit separation given other params, if a0_mode=Roche_limit_fraction
        
        Rdfunc=None,            # donor-star radius function:  Rd/Rd0 = Rdfunc(Md/Md0) 
        
        fcorot0=1.0,            # initial degree of synchronization, range 0-1. 
        
        gamma_adiabatic=5./3.,  # equation of state of donor, gamma_adiabatic , along an adiabat: P \propto \rho^gamma_adiabatic
        
        gamma_structure=5./3.   # structural gamma of the donor star (polytropic gamma): gamma_structure=1+1/n
        
        fcorot0=1.0,             # initial degree of binary corotation
        
        gamma_adiabatic=5./3.,   # gas equation of state: adiabatic index
        
        gamma_structure=5./3.    # polytropic index of the donor gamma_structure = 1 + 1/n
        
        Mdtot = None,            # original total mass of the donor (if Md0 is intended to be less than the original donor mass, as in starting part way through the donor mass radius relationship. 

        verbose = True           # print information? 
        """
        
        self.c = Constants()
        self.t0 = 0.0
        self.Md0 = Md0
        self.Ma0 = Ma0
        self.Rd0 = Rd0
        self.Ra0 = Ra0
        self.G  = Ggrav
        self.Mtot0 = self.Md0 + self.Ma0
        
        if a0_mode == 'manual':
            self.a0  = a0
        elif a0_mode == 'Roche_limit_fraction':
            self.a0 = f_roche_limit * self.a_o_RL(self.Ma0/self.Md0) * self.Rd0
        elif a0_mode == 'period':
            self.a0 = (self.G*self.Mtot0 *(P0/(2*np.pi))**2 )**(1./3.)
            
        if Rdfunc==None:
            self.Rdfunc = self.Rdfunc_constant
        else:
            self.Rdfunc = Rdfunc
        
        self.fcorot = fcorot0
        self.gamma_adiabatic = gamma_adiabatic
        self.gamma_structure = gamma_structure


        if amin_integrate == None:
            self.amin = self.Rd0
        else:
            self.amin = amin_integrate

        if Mdtot == None:
            self.Mdtot = Md0
        else:
            self.Mdtot = Mdtot

        if(verbose==True):
            print("=== RLOF: binary defined =======")
            print("Md0 = ",self.Md0)
            print("Ma0 = ",self.Ma0)
            print("Rd0 = ",self.Rd0)
            print("Ra0 = ",self.Ra0)
            print("a0 = ",self.a0)
            print("G = ",self.G)
            print("----------donor star------------")
            print("Md0 = ",self.Md0)
            print("Md(original) = ",self.Mdtot)
            print("Rd0 = ",self.Rd0)
            print("Rdfunc = ",self.Rdfunc)
            print("fcorot0 = ",self.fcorot)
            print("gamma_adiabatic = ",self.gamma_adiabatic)
            print("gamma_structure = ",self.gamma_structure)
            print("================================")
        
            
    def Rdfunc_constant(self,Mdonor_over_Mdtot):
        """ 
        default version of donor radius (R/R0) as a function of donor mass (M/M0)
            R/R0 = f(M/M0)
        """
        Rd_over_Rd0 = 1.0
        return Rd_over_Rd0
        
        
    def a_o_RL(self,q):
        """Eggelton formula, q=Ma/Md (opposite of eggelton deff) 
        this is: a/rL where a is separation and rL is the Roche Lobe radius (around the donor)"""
        return (0.6*q**(-2./3.) + np.log(1+q**(-1./3.)))/(0.49*q**(-2./3.))

        
    def p_orb(self,Mtot,a):
        """Orbital period"""
        return (2*np.pi*np.sqrt(a**3/(self.G*Mtot)))
    
    
    def mdot_donor(self,a,md,ma,rd,
                   fcorot=1.,gad=5./3.,gs=5./3.,
                   mdot_mode='simulation',alpha_manual=1.0):
        """
        Mdot from the donor star, includes the option to set 
        the normalization either via the simulation interpolation 
        or manually. 
        Pols eqn 7.5
        """
        rL = self.a_o_RL(ma/md)**-1 * a
        if mdot_mode=='simulation':
            alpha = self.approx_alpha(ma/md,fcorot,gad,gs)
        elif mdot_mode=='manual':
            alpha = alpha_manual
        else:
            print("MDOT MODE NOT RECOGNIZED")
            
        mdot = -alpha*md/self.p_orb(md+ma,a)*((rd-rL)/rd)**((3.*gs-1)/(2*gs-2))
        
        if mdot<0:
            return mdot
        else:
            return 0.0
    
    def approx_gamma(self,q, fcorot, gad, gs):
        """approximation of gamma_loss"""
        gloss_norm = 0.66*(q/0.1)**0.08 * (gad/(5/3.))**0.69 * (gs/(5/3.))**-2.17  * (1 - 0.30*(fcorot-1.) )
        gamma_donor = q
        gamma_L2 = (1+q)**2/q * 1.2**2
        return (gamma_L2-gamma_donor)*gloss_norm + gamma_donor
    
    def approx_alpha(self,q, fcorot, gad, gs):
        """approximation of alpha_mdot"""
        alpha = 0.62*(q/0.1)**0.68 * (gad/(5/3.))**5.39 * (gs/(5/3.))**-3.25  * (1 - 0.89*(fcorot-1.) )
        return alpha
    
    
    def derivs(self,t,vec,
           loss_mode="simulation",
           mdot_mode='simulation',
           acc_mode="manual",
           beta_manual=0.0,
           alpha_manual=1.0):

        """Pols Ch 7, Lajoie&Sills 2011
        vector:
        md = mass of donor
        ma = mass of accretor
        a = separation

        params:
        t = time

        loss_mode = donor, accretor, l2 -- specific angular momentum of material lost
        acc_modt = "Eddington, manual" -- eddington assumes cgs units
        """

        # unpack    
        md,ma,a = vec

        # loss of angular momentum
        if loss_mode=='donor':
            gamma = ma/md  # pols
        elif loss_mode=='accretor':
            gamma = md/ma  # pols
        elif loss_mode=='l2':
            gamma = (md+ma)**2/(md*ma) * 1.2**2  # pribula 
        elif loss_mode=='simulation':
            gamma = self.approx_gamma(ma/md, self.fcorot, self.gamma_adiabatic, self.gamma_structure)
        else:
            print("INVALID loss_mode")


        # derivatives
        
        # donor
        # set donor radius
        rd = self.Rdfunc(md/self.Mdtot)*self.Rd0
        # set mdot
        dmddt = self.mdot_donor(a,md,ma,rd,
                                self.fcorot,self.gamma_adiabatic, self.gamma_structure,
                                mdot_mode=mdot_mode,alpha_manual=alpha_manual)

        # accretion fraction
        if acc_mode=='eddington':
            Leddington = 1.26e38*u.erg/u.s*(ma/c.M_sun.cgs)
            dmdt_edd = Leddington*ra/(self.G*ma)
            beta = max(1.0,dmdt_edd/dmddt)
        elif acc_mode=='manual':
            beta = beta_manual
        else:
            print( "INVALID acc_mode")
            
        dmadt = -beta*dmddt
        
        # orbit
        dadt = -2.0*a*dmddt/md*(1.0 - beta*md/ma - (1.0-beta)*(gamma+0.5)*md/(md+ma))

        return np.array( (dmddt,dmadt,dadt) )
    
    
    def __event_CE(self,t,vec):
        md,ma,a = vec
        return self.amin - a
    __event_CE.terminal = True
    
    
    def integrate(self,dt,
                  Ntimes=101,
                  loss_mode="simulation",
                  mdot_mode='simulation',
                  acc_mode="manual",
                  beta_manual=0.0,
                  alpha_manual=1.0,
                  rtol=1.e-8,
                  return_ivp=False,
                  verbose=True):
        """ 
        Integrate the solution forward from the initial conditions to time = dt, saving Ntimes outputs. 
        Returns an astropy Table.
        
        Parameters:
        dt,                      # time to integrate to 
        Ntimes=101,              # number of outputs to save
        loss_mode="simulation",  # specific angular momentum loss mode, options: donor, accretor, l2, simulation
        mdot_mode='simulation',  # normalization of mdot_donor, options: manual, simulation
        acc_mode="manual",       # mode of accretion by the accretor, options: manual, eddington
        beta_manual=0.0,         # if acc_mode='manual', this is the fraction accreted by the accretor
        alpha_manual=1.0,        # if mdot_mode='manual', this is the normalization factor alpha
        return_ivp=False,        # return the full ivp output from solve_ivp
        verbose = True           # print diagnostic output
        """
        
        ic = (self.Md0,self.Ma0,self.a0)
        times = np.linspace(self.t0,self.t0+dt,Ntimes)
        
        ivp = solve_ivp(fun=lambda t, y: self.derivs(t, y,loss_mode,mdot_mode,acc_mode,beta_manual,alpha_manual),
                        t_span=(times[0],times[-1]),
                        y0=np.array(ic),
                        t_eval=times,
                        events=(self.__event_CE),
                        method='BDF',
                        rtol=rtol,
                        atol=0.0)

        if (verbose==True):
            print ("---- integration ---------------")
            print ("solver message: ",ivp.message)
            print ("events: ",ivp.t_events)
            print ("--------------------------------")
        
        solT = Table(data=ivp.y.T,names=['Md','Ma','a'])
        solT['t'] = ivp.t
        solT['Rd'] = self.Rdfunc(solT['Md']/self.Mdtot) * self.Rd0
        solT['Ra'] = self.Ra0
        solT['dMddt']=np.gradient(solT['Md'])/np.gradient(solT['t'])

        if return_ivp==False:
            return solT[['t','Md','Rd','Ma','Ra','a','dMddt']]
        else:
            return solT[['t','Md','Rd','Ma','Ra','a','dMddt']], ivp



    
    def rho_torus(self,R,z,K=1.,gamma=4./3.,l=2.,GM=1.,R0=100.):
        """
        definition of a constant angular momentum torus in HSE. parameters:
        R,z cylindrical coords
        K,gamma: P=K*rho^gamma, polytropic eos for the torus gas  
        l = specific angular momentum 
        GM = central mass*G
        R0 = R at which density vanishes (outer radius)
        """
        r = np.sqrt(R**2 + z**2)
        dens = ((gamma-1.)/(K*gamma) * (GM/r - l**2/(2*R**2) - GM/R0 + l**2/(2*R0**2)))**(1./(gamma-1))
        return np.nan_to_num(dens)
