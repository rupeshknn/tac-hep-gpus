#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "t1.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h> 
#include <TLorentzVector.h>

using namespace std;

//------------------------------------------------------------------------------
// Particle Class
//
class Particle{

	public:
	Particle();
	// FIXME : Create an additional constructor that takes 4 arguments --> the 4-momentum
	Particle (double, double, double, double);
	double   pt, eta, phi, E, m, p[4];
	void     p4(double, double, double, double);
	void     print();
	void     setMass(double);
	double   sintheta();
};

//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle(){
	pt = eta = phi = E = m = 0.0;
	p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor ------------------------------------------------------
Particle::Particle(double p0, double p1, double p2, double p3){
	//FIXME
	TLorentzVector part;
	part.SetXYZT(p1,p2,p3,p0);

	this->p[0] = part[0];
	this->p[1] = part[1];
	this->p[2] = part[2];
	this->p[3] = part[3];
	this->pt = part.Pt();
	this->eta = part.Eta();
	this->phi = part.Phi();
	this->E = part.E();
	this->m = part.M();
}

//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta(){
	//FIXME
	TLorentzVector particle;
	particle.SetXYZT(this->p[1],this->p[2],this->p[3],this->p[0]);
	return sin(particle.Theta());
}

void Particle::p4(double pT, double eta, double phi, double energy){
	// FIXME
	TLorentzVector particle;
	particle.SetPtEtaPhiE(pT, eta, phi, energy);
	this->p[0] = particle[0];
	this->p[1] = particle[1];
	this->p[2] = particle[2];
	this->p[3] = particle[3];
}

void Particle::setMass(double mass)
{
	// FIXME
	this->m = mass;
}

//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print(){
	std::cout << std::endl;
	std::cout << "(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << std::endl;
}


class Lepton : public Particle {
	using Particle::Particle;
	public:
	signed int	Q;
	void set_charge(signed int charge){
		this->Q = charge;
	};
};

class Jet : public Particle {
	using Particle::Particle;
	public:
	int	f;
	void set_flavor(int flavor){
		this->f = flavor;
	};
};

int main() {
	
	/* ************* */
	/* Input Tree   */
	/* ************* */

	TFile *f      = new TFile("input.root","READ");
	TTree *t1 = (TTree*)(f->Get("t1"));

	// Read the variables from the ROOT tree branches
	t1->SetBranchAddress("lepPt",&lepPt);
	t1->SetBranchAddress("lepEta",&lepEta);
	t1->SetBranchAddress("lepPhi",&lepPhi);
	t1->SetBranchAddress("lepE",&lepE);
	t1->SetBranchAddress("lepQ",&lepQ);
	
	t1->SetBranchAddress("njets",&njets); // not defined in t1.h
	t1->SetBranchAddress("jetPt",&jetPt);
	t1->SetBranchAddress("jetEta",&jetEta);
	t1->SetBranchAddress("jetPhi",&jetPhi);
	t1->SetBranchAddress("jetE", &jetE);
	t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);

	// Total number of events in ROOT tree
	Long64_t nentries = t1->GetEntries();

	for (Long64_t jentry=0; jentry<100;jentry++)
 	{
		t1->GetEntry(jentry);
		std::cout<<" Event "<< jentry <<std::endl;	

		//FIX ME
		// cout << njets << ", " << sizeof(jetE)<< ", " << sizeof(lepE)<< ", " << endl;
		for (Long_t part=0; part<sizeof(jetE); part++)
		{
			cout << " Jet particles" << endl;
			Jet jet_object;
			jet_object.p4(jetPt[part], jetEta[part], jetPhi[part], jetE[part]);
			jet_object.set_flavor(jetHadronFlavour[part]);
			jet_object.print();

			cout << " Lepton particles" << endl;
			Lepton lepton_object;
			lepton_object.p4(lepPt[part], lepEta[part], lepPhi[part], lepE[part]);
			lepton_object.set_charge(lepQ[part]);
			lepton_object.print();
		}

	} // Loop over all events

  	return 0;
}
