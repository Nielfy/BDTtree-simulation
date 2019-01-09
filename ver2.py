# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 08:51:23 2018

@author: wanly
"""
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import fsolve
def InstantR(DD,DeltaT):
    InstantR=[]
    for i in range(len(DD)):
        if i==0:
            InstantR.append((1-DD[i])/(DeltaT*DD[i]))
            continue
        InstantR.append((DD[i-1]-DD[i])/(DeltaT*DD[i]))
    return InstantR

def GenerateV(B,L,DeltaT,K,Sigma=0.05):
    n=len(L)-1
    V=[]
    for i in range(n):
        if i==0:
            temp=0
            V.append(temp)
            continue
        T1=DeltaT*i
        d1=(np.log(L[i+1]/K[i+1])+0.5*np.power(Sigma,2)*T1)/(Sigma*np.power(T1,0.5))
        d2=d1-Sigma*np.power(T1,0.5)
        temp=B[i+1]*(L[i+1]*scipy.stats.norm(0,1).cdf(d1)-K[i+1]*scipy.stats.norm(0,1).cdf(d2))
        V.append(temp)
    return V

def Tree(Years,DeltaT,V,p,K):
    n=(len(p)-1)
    if n!=int(Years/DeltaT):
        print("Error")
    bdt=[]
    debrew=[]
    u=[]
    Sigma=[]
    for i in range(n+1):
        if i==0:
            #bdt.append(list((p[0]-p[1])/(DeltaT*p[1])))
            debrew.append(list([1]))
            
            #u.append(bdt[0])
            continue
        x=least_squares(lambda u:Equation(r(u[0],u[1],i,DeltaT),D(debrew[-1],r(u[0],u[1],i,DeltaT),i,DeltaT),debrew[-1],p[i],V[i-1],K[i]),[0.05,0.05],bounds=([0,0],[1,1]),ftol=1e-16, xtol=1e-16, gtol=1e-16)
        uu=x.x[0]
        ss=x.x[1]
        u.append(uu)
        Sigma.append(ss)
        bdt.append(r(uu,ss,i,DeltaT))
        debrew.append(D(debrew[i-1],bdt[-1],i,DeltaT))
    return bdt,debrew,u,Sigma

def r(x,y,i,DeltaT): 
    rr=[]
    for j in range(i):
        temp=x*np.exp((i-1-j*2)*y*np.power(DeltaT,0.5))
        rr.append(temp)
    return rr

def D(Deb,rr,i,DeltaT):
    dd=[]
    for j in range(i+1):
        if j==0:
            temp=0.5*Deb[0]*np.exp(-1*rr[0]*DeltaT)
            dd.append(temp)
            continue
        if j==i:
            temp=0.5*Deb[-1]*np.exp(-1*rr[-1]*DeltaT)
            dd.append(temp)
            continue
        temp=0.5*Deb[j-1]*np.exp(-1*rr[j-1]*DeltaT)+0.5*Deb[j]*np.exp(-1*rr[j]*DeltaT)
        dd.append(temp)
    return dd
    
def Equation(rr,dd,Deb,p,V,K):
    Eq1=np.sum(dd)-p
    temp=0
    for i in range(len(rr)):
        temp=temp+max(rr[i]-K,0)*np.exp(-1*rr[i]*DeltaT)*Deb[i]
        
    Eq2=temp-V
    return [Eq1,Eq2]
        
def Cappricer(bdt,deb,t,DeltaT,K):
    i=int(t/DeltaT)
    temp=0
    for m in range(i+1):
        temp=temp+max(bdt[i][m]-K,0)*np.exp(-1*bdt[i][m]*DeltaT)*deb[i][m]
    return temp

def FRApricer(bdt,deb,t,DeltaT):
    def FRAhelper(bdt,deb,t,DeltaT,K):
        i=int(t/DeltaT)
        temp=0
        pay=[]
        for m in range(i+1):
            temp=temp+(bdt[i][m]-K)*np.exp(-1*bdt[i][m]*DeltaT)*deb[i][m]
        return temp
    kk=fsolve(lambda K:FRAhelper(bdt,deb,t,DeltaT,K),0.01)
    return float(kk)

def prepaidFRA(bdt,deb,t,DeltaT):
    def FRAhelper(bdt,deb,t,DeltaT,K):
        i=int(t/DeltaT)
        temp=0
        for m in range(i+1):
            temp=temp+(bdt[i][m]-K)*DeltaT*deb[i][m]
        return temp
    kk=fsolve(lambda K:FRAhelper(bdt,deb,t,DeltaT,K),0.01)
    return float(kk)
if __name__ == '__main__':
    Years=10 #years = n*DeltaT,n=len(pp)-1
    DeltaT=0.25
    #p 0->n n+1
    #L i-1->i n
    pp=[1.0,
     0.9879431223996096,
     0.97603161309669,
     0.9643901397323486,
     0.9528889219315109,
     0.9407491824814741,
     0.9287284991476754,
     0.9169215513561094,
     0.905268946682189,
     0.8934232530845351,
     0.8816951593083274,
     0.8701761741452341,
     0.8588152591667025,
     0.8471756434359966,
     0.8356162069917865,
     0.8242653738685239,
     0.813079579798389,
     0.8016551396647688,
     0.7902882371672962,
     0.7791291219562688,
     0.7681415636042237,
     0.7573391073936924,
     0.746699377609058,
     0.7362496006068493,
     0.725962278590809,
     0.7157811881100277,
     0.7057252985460726,
     0.695845949555491,
     0.6861231716405903,
     0.676395538964387,
     0.6667216173805769,
     0.6572170940632445,
     0.6478684628141306,
     0.6386243241013684,
     0.6294906127195203,
     0.6205141264238461,
     0.6116875791456277,
     0.6029623257988445,
     0.5943386582526324,
     0.5858608739329235,
     0.5775272543068408]
    
    L=InstantR(pp,DeltaT)
    K=L
    V=GenerateV(pp,L,DeltaT,K)
    
    bdt,debrew,u,Sigma=Tree(Years,DeltaT,V,pp,K)
    for i in range(40):
        print(Cappricer(bdt,debrew,0.25*i,0.25,K[i+1])-V[i])
        
    for i in range(len(debrew)):
        print(sum(debrew[i])-pp[i])

    
    print(prepaidFRA(bdt,debrew,1,0.25),FRApricer(bdt,debrew,1,0.25))
    plt.plot(pp)
    plt.show()
    plt.plot(K)
    plt.show()
    plt.plot(V)
    plt.show()
    plt.plot(Sigma)
    plt.show()
    plt.plot(u)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    