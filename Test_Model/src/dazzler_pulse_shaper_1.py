#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import re
import math
import matplotlib.pyplot as plt
import scipy.fftpack
import pylab
from scipy.interpolate import interp1d
"""
Created on Sat Oct  3 13:38:57 2020

@author: jackhirschman
"""

'''

class Pulse_Shaper():
    
    
    def __init__(self, wavelength, bandwidth, crystal_length, characteristic_acoustic_power, standard_geometry="HR"):
        self.P0 = characteristic_acoustic_power
        self.L = crystal_length
        self.lambd = wavelength
        self.bw = bandwidth
        self.standard_geometry = standard_geometry
        self.del_lamb = 8.9*self.lambd**2/self.L # equation 2.16
        self.N = self.bw/self.del_lamb #equation 2.17
        self.ang_divergence = 22*self.lambd/self.L #equation 2.18
        
        
    def fft_Efield(self, electric_field):
        return np.fft(electric_field)
    
    def get_diffraction_efficiency(self, k_diff, k_in, k_ac):
        u_ac = k_ac/np.sqrt(np.sum(k_ac**2))
        return np.dot(k_diff - k_in - k_ac, u_ac) 
      
    def convolv_Efield_transferFunc(self, electric_field, transferFunc):
        return electric_field*transferFunc
    
    def get_transfer_function(self, eta, spectral_phase):
'''
'''
        Takes eta (diffraction efficiency) and spectral phase

        Parameters
        ----------
        eta : TYPE
            DESCRIPTION.
        spectral_phase : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
'''
'''
        return np.sqrt(eta)*np.exp(1j*spectral_phase)
    def get_diffraction_efficiency(self, power):
        delta_k = self.get_phase_mismatch(k_diff, k_in, k_ac)
        return np.pi**2/4 * power/self.P0 * (np.sinc(np.sqrt(np.pi**2/4 * power/self.P0 + (delta_k*self.L/2)**2)))**2
 '''   
class Pulse_Shaper_Dial():
    
    
    def __init__(self, position, width, hole_position, hole_width, hole_depth,
                 delay, sec_order, third_order, fourth_order, c = 299792458.0):
        self.position = position
        self.width = width
        self.hole_position = hole_position
        self.hole_depth = hole_depth
        self.hole_width = hole_width
        
        self.delay = delay
        self.sec_order = sec_order
        self.third_order = third_order
        self.fourth_order = fourth_order
        
        self.c = c
        
    def calculate_parameters(self, c = 3*10**8):
        omega0 = 2*np.pi*c/self.position
        chi0 = self.width/(2*self.position)
        del_omega0 = omega0*(chi0-chi0**3)
        
        omega1 = 2*np.pi*c/self.hole_position
        chi1 = self.hole_width/(2*self.hole_position)
        del_omega1 = omega1*(chi1-chi1**3)/2
        
        return omega0, chi0, del_omega0, omega1, chi1, del_omega1
    
    def calculate_amplitude_transfer_function(self, ang_freq_vector, components_return = False):
        omega0, chi0, del_omega0, omega1, chi1, del_omega1 = self.calculate_parameters()
        #f =(np.exp(-(ang_freq_vector-omega0)/del_omega0))**6
        f =(np.exp(-((ang_freq_vector-omega0)/del_omega0)**6))
        #g = 1 - self.hole_depth*(np.exp(-(ang_freq_vector-omega1)/del_omega1))**2
        g = 1 - self.hole_depth*(np.exp(-((ang_freq_vector-omega1)/del_omega1)**2))

        
        if (components_return):
            return f, g, f*g
        return f*g
    
    def calculate_phase_transfer_function(self, ang_freq_vector):
        omega0, chi0, del_omega0, omega1, chi1, del_omega1 = self.calculate_parameters()
        omega_dif = ang_freq_vector-omega0
        return -(self.delay*omega_dif + self.sec_order/2 * omega_dif**2 + self.third_order/6 * omega_dif**3 + 
                 self.fourth_order/24 * omega_dif**4)
    
    def calculate_full_transfer_function(self,ang_freq_vector, S_saved=0, a_saved=0, components_return=False):
        f,g,A_dial = None,None, None
        if (components_return):
            f,g,A_dial = self.calculate_amplitude_transfer_function(ang_freq_vector, components_return=True)
        else:
            A_dial = self.calculate_amplitude_transfer_function(ang_freq_vector, components_return=False)
            
        phi_dial = self.calculate_phase_transfer_function(ang_freq_vector)
        
        S_full = A_dial*np.exp(1j*phi_dial) + a_saved*np.exp(1j*phi_dial)*S_saved
        
        if (components_return):
            return f,g,A_dial,phi_dial, S_full
        
        return S_full
        
    def shape_input_pulse(self, E_field_input, time_vector, sampling_rate, S_saved=0, a_saved=0, generate_plots=False, components_return=False):
        E_field_input_ft = np.fft.fft(E_field_input)
        freq_vector = np.fft.fftfreq(n=E_field_input_ft.size, d=1/sampling_rate)
        ang_freq_vector = 2*np.pi*freq_vector
        
        f,g,A_dial,phi_dial, S_full = None, None, None, None, None
        if (components_return):
            f,g,A_dial,phi_dial, S_full = self.calculate_full_transfer_function(ang_freq_vector, S_saved, a_saved, components_return)
            components_dict = {"f":f,"g":g,"A_dial":A_dial,"phi_dial":phi_dial,"S_full":S_full}

        else :
            S_full = self.calculate_full_transfer_function(ang_freq_vector, S_saved, a_saved, components_return)
            components_dict = {"S_full":S_full}
        
        E_field_output_ft = E_field_input_ft*S_full
        E_field_output = np.fft.ifft(E_field_output_ft)
        
        #wave_vec = self.c/freq_vector
        wave_vec = np.linspace(10e-9, 1300e-9, num = len(time_vector))
        
        
        #Handling input
        #E_field_input_wavelength_ft = (E_field_input_ft)
        intensity_input = (np.abs(E_field_input))**2
        phase_input = -1*np.arctan(np.imag(E_field_input)/np.real(E_field_input))
        spectrum_freq_input = (np.abs(E_field_input_ft))**2
        spectrum_freq_input_interp = interp1d(freq_vector, spectrum_freq_input)
        #spectrum_wavelength_input = (2*np.pi*self.c/(wave_vec**2)) * (np.abs(E_field_input_wavelength_ft))**2
        spectrum_wavelength_input = (2*np.pi*self.c/(wave_vec**2)) *spectrum_freq_input_interp(self.c/wave_vec)
        spectral_phase_freq_input = -1*np.arctan(np.imag(E_field_input_ft)/np.real(E_field_input_ft))
        spectral_phase_freq_input_interp = interp1d(freq_vector,spectral_phase_freq_input)
        spectral_phase_wavelength_input = spectral_phase_freq_input_interp(self.c/wave_vec)
        #spectral_phase_wavelength_input = -1*np.arctan(np.imag(E_field_input_wavelength_ft)/np.real(E_field_input_wavelength_ft))


        #Handling output
        #E_field_output_wavelength_ft=(E_field_output_ft)
        intensity_output = (np.abs(E_field_output))**2
        phase_output = -1*np.arctan(np.imag(E_field_output)/np.real(E_field_output))
        spectrum_freq_output = (np.abs(E_field_output_ft))**2
        spectrum_freq_output_interp = interp1d(freq_vector, spectrum_freq_output)
        spectrum_wavelength_output = (2*np.pi*self.c/(wave_vec**2)) *spectrum_freq_output_interp(self.c/wave_vec)
        #spectrum_wavelength_output = (2*np.pi*self.c/(wave_vec**2)) * spectrum_freq_output
        spectral_phase_freq_output = -1*np.arctan(np.imag(E_field_output_ft)/np.real(E_field_output_ft))
        spectral_phase_freq_output_interp = interp1d(freq_vector, spectral_phase_freq_output)
        spectral_phase_wavelength_output = spectral_phase_freq_output_interp(self.c/wave_vec)
        #spectral_phase_wavelength_output = -1*np.arctan(np.imag(E_field_output_wavelength_ft)/np.real(E_field_output_wavelength_ft))
 
        #Handling transfer function
        S_full_time = np.fft.ifft(S_full)
        intensity_transfer = (np.abs(S_full_time))**2
        phase_transfer= -1*np.arctan(np.imag(S_full_time)/np.real(S_full_time))
        spectrum_freq_transfer = (np.abs(S_full))**2
        spectrum_freq_transfer_interp = interp1d(freq_vector, spectrum_freq_transfer)
        spectrum_wavelength_transfer = (2*np.pi*self.c/(wave_vec**2)) *spectrum_freq_transfer_interp(self.c/wave_vec)
        #spectrum_wavelength_transfer = (2*np.pi*self.c/(wave_vec**2)) * spectrum_freq_transfer
        spectral_phase_freq_transfer = -1*np.arctan(np.imag(S_full)/np.real(S_full))
        spectral_phase_freq_transfer_interp = interp1d(freq_vector, spectral_phase_freq_transfer)
        spectral_phase_wavelength_transfer = spectral_phase_freq_transfer_interp(self.c/wave_vec)
        #spectral_phase_wavelength_transfer = spectral_phase_freq_transfer

        
        if (generate_plots):
            plot_list = [E_field_input, E_field_output, E_field_input_ft, E_field_output_ft]
            plot_list_labels = ["E Field Input", "E Field Output"]
            titles = ["Intensity", "Phase","Intensity", "Phase","Intensity", "Phase"]
            plot_list_domains = [time_vector, freq_vector]
            
            self.standard_plot_pattern(time_vector, intensity_input,phase_input,
                                  intensity_transfer, phase_transfer, intensity_output,phase_output,
                                  "Input, Transfer, Output", titles, "time (ps)", "arb", x_scale = 1e-15,
                                  show=False,param_box=True, position=self.position, width=self.width,
                                  hole_position=self.hole_position, hole_width=self.hole_width,
                                  hole_depth=self.hole_depth, delay=self.delay, sec_order=self.sec_order,
                                  third_order=self.third_order, fourth_order=self.fourth_order)
            #self.standard_plot_pattern(np.fft.fftshift(wave_vec), np.fft.fftshift(spectrum_wavelength_input),np.fft.fftshift(spectral_phase_wavelength_input),
                                  #np.fft.fftshift(spectrum_wavelength_transfer), np.fft.fftshift(spectral_phase_wavelength_transfer), np.fft.fftshift(spectrum_wavelength_output),
                                  #np.fft.fftshift(spectral_phase_wavelength_output),
                                  #"Input, Transfer, Output", titles, "wavelength (nm)", "arb", x_scale = 1e-9,
                                  #show=False,param_box=True, position=self.position, width=self.width,
                                  #hole_position=self.hole_position, hole_width=self.hole_width,
                                  #hole_depth=self.hole_depth, delay=self.delay, sec_order=self.sec_order,
                                  #third_order=self.third_order, fourth_order=self.fourth_order)
            self.standard_plot_pattern(wave_vec, spectrum_wavelength_input,spectral_phase_wavelength_input,
                                  spectrum_wavelength_transfer, spectral_phase_wavelength_transfer, spectrum_wavelength_output,
                                  spectral_phase_wavelength_output,
                                  "Input, Transfer, Output", titles, "wavelength (nm)", "arb", x_scale = 1e-9,
                                  show=False,param_box=True, position=self.position, width=self.width,
                                  hole_position=self.hole_position, hole_width=self.hole_width,
                                  hole_depth=self.hole_depth, delay=self.delay, sec_order=self.sec_order,
                                  third_order=self.third_order, fourth_order=self.fourth_order)
            
            self.standard_plot_pattern(np.fft.fftshift(freq_vector), np.fft.fftshift(spectrum_freq_input),np.fft.fftshift(spectral_phase_freq_input),
                                  np.fft.fftshift(spectrum_freq_transfer), np.fft.fftshift(spectral_phase_freq_transfer), np.fft.fftshift(spectrum_freq_output),
                                  np.fft.fftshift(spectral_phase_freq_output),
                                  "Input, Transfer, Output", titles, "freq (Hz)", "arb", x_scale = 1)
            
            
            '''
            self.plot(plot_list[0:2], plot_list_domains[0], x_axis_label = "time (ps)", x_scaling = 1e-12,
                      y_axis_label="Electric Field", labels = plot_list_labels, spacing = "linear",normalize=True,  multi_plots=True, 
                      require_fft_shift=False)
            self.plot(plot_list[2:], plot_list_domains[1], x_axis_label = "frequency (Hz)", x_scaling = 1,
                      y_axis_label="Electric Field", labels = plot_list_labels, spacing = "linear",normalize=True,  multi_plots=True, 
                      require_fft_shift=True)
            #self.plot(np.fft.fftshift(plot_list[2:]), 3e8/((plot_list_domains[1])), x_axis_label = "wavelength (nm)", x_scaling = 1e-9,
                      #y_axis_label="Electric Field", labels = plot_list_labels, spacing = "linear",normalize=True,  multi_plots=True, 
                      #require_fft_shift=False)
            self.plot(np.abs(np.fft.fftshift(plot_list[3])), 3e8/(np.fft.fftshift(plot_list_domains[1])), x_axis_label = "wavelength (nm)", x_scaling = 1e-9,
                      y_axis_label="Electric Field", labels = plot_list_labels[1], spacing = "linear",normalize=True,  multi_plots=False, 
                      require_fft_shift=False)
            '''
        return E_field_output,time_vector, E_field_output_ft, freq_vector, components_dict
    
    def plot(self, y_data, x_data, x_axis_label, x_scaling, y_axis_label, labels, spacing, normalize=False,
             multi_plots = False, require_fft_shift=False):
        x_data_plot = x_data/x_scaling
        y_data_plot = []
        if (normalize):
            if (multi_plots):
                for yd in y_data:
                    y_data_plot.append(yd/np.linalg.norm(yd))
            else:
                y_data_plot.append(y_data/np.linalg.norm(y_data))
        else:
            if (multi_plots):
                y_data_plot = y_data
            else:
                y_data_plot.append(y_data)
        
        fig, axs = plt.subplots()
        ii = 0
        for yd in y_data_plot:
            yd_plot = None
            xd_plot = None
            if (require_fft_shift):
                yd_plot = np.abs(np.fft.fftshift(yd))
                xd_plot = np.fft.fftshift(x_data_plot)
            else:
                yd_plot = yd
                xd_plot = x_data_plot
            axs.plot(xd_plot, yd_plot, label=labels[ii])
            ii +=1
        axs.legend()
        axs.set_xlabel(x_axis_label)
        axs.set_ylabel(y_axis_label)
        
        if (spacing=="ylog"):
            axs.set_yscale('log')
        elif (spacing == "loglog"):
            axs.set_yscale('log')
            axs.set_xscale('log')
            
            
        
        plt.show()
        
        return
    
    def standard_plot_pattern(self, xvec, input_data_amp,input_data_phase, 
                              transfer_fnc_amp,transfer_fnc_phase,output_data_amp,
                              output_data_phase,plot_title,titles, xaxis_label, yaxis_label,
                              x_scale = 1, show=True,param_box=False, position=None, width=None,hole_position=None,
                              hole_width=None,hole_depth=None, delay=None, sec_order=None, third_order=None,
                              fourth_order=None, phase_unwrap = True):
        '''
        #trying to fix phase unwrapping
        
        epsilon = 1e-5
        if (phase_unwrap):
            for ii in range(1,input_data_phase.shape[0]):
                if (np.abs(input_data_phase[ii] - input_data_phase[ii-1] -np.pi) <= epsilon):
                    input_data_phase[ii] -= np.pi
                elif (np.abs(input_data_phase[ii] - input_data_phase[ii-1] +np.pi) <= epsilon):
                    input_data_phase[ii] += np.pi
                    
            for ii in range(1,transfer_fnc_phase.shape[0]):
                if (np.abs(transfer_fnc_phase[ii] - transfer_fnc_phase[ii-1] -np.pi) <= epsilon):
                    transfer_fnc_phase[ii] -= np.pi
                elif (np.abs(transfer_fnc_phase[ii] - transfer_fnc_phase[ii-1] +np.pi) <= epsilon):
                    transfer_fnc_phase[ii] += np.pi
                    
            for ii in range(1,output_data_phase.shape[0]):
                if (np.abs(output_data_phase[ii] - output_data_phase[ii-1] -2*np.pi) <= epsilon):
                    output_data_phase[ii] -= 2*np.pi
                elif (np.abs(output_data_phase[ii] - output_data_phase[ii-1] +2*np.pi) <= epsilon):
                    output_data_phase[ii] += 2*np.pi
        '''
        xvec=xvec/x_scale
        fig, axs = plt.subplots(2,3)
        fig.suptitle(plot_title)
        axs[0,0].plot(xvec,input_data_amp)
        axs[0,0].set_title(titles[0])
        axs[1,0].plot(xvec, input_data_phase,'tab:green')
        axs[1,0].set_title(titles[1])
        axs[0,1].plot(xvec,transfer_fnc_amp,'tab:orange')
        axs[0,1].set_title(titles[2])
        axs[1,1].plot(xvec, transfer_fnc_phase,'tab:red')
        axs[1,1].set_title(titles[3])
        axs[0,2].plot(xvec,output_data_amp)
        axs[0,2].set_title(titles[4])
        axs[1,2].plot(xvec, output_data_phase,'tab:green')
        axs[1,2].set_title(titles[5])
        
        for ax in axs.flat:
            ax.set(xlabel=xaxis_label,ylabel=yaxis_label)
        
            
        # Adjust spacing
        left = 0.1  # the left side of the subplots of the figure
        right = 0.97   # the right side of the subplots of the figure
        bottom = 0.08  # the bottom of the subplots of the figure
        top = 0.92     # the top of the subplots of the figure
        wspace = 0.3  # the amount of width reserved for space between subplots,
                      # expressed as a fraction of the average axis width
        hspace = 0.3  # the amount of height reserved for space between subplots,
                      # expressed as a fraction of the average axis height
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)  
        
        if (param_box):
            textstr_params = '\n'.join((
                r'$\mathrm{position}= %.2f \mathrm{nm}$' % (position/1e-9, ),
                r'$\mathrm{width}= %.2f \mathrm{nm}$' % (width/1e-9, ),
                r'$\mathrm{hole position}= %.2f \mathrm{nm}$' %(hole_position/1e-9, ),
                r'$\mathrm{hole width}= %.2f \mathrm{nm}$' %(hole_width/1e-9, ),
                r'$\mathrm{hole depth}= %.2f$' %(hole_depth, ),
                r'$\mathrm{delay}= %2f$' %(delay, ),
                r'$\mathrm{second order}= %.2f$' %(sec_order, ),
                r'$\mathrm{third order}= %.2f$' %(third_order, ),
                r'$\mathrm{fourth order}= %.2f$' %(fourth_order, ),
                ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(.95, .95, textstr_params, transform=ax.transAxes, fontsize = 14,
                     verticalalignment='top', bbox=props)
        
        if (show):
            plt.show()
        
        return
        
        
        
        
    
    
def main():
    position = 800e-9
    width = 400e-9#93.9e-9
    hole_position = 850e-9
    hole_width = 300e-9#3e-9
    hole_depth=1300
    delay=1e-14#4250e-15
    sec_order=22000*(1e-15)**2
    third_order=10*(1e-15)**3
    fourth_order=0*(1e-15)**4
    c = 299792458.0 # m/s
    time_vector = np.linspace(-10e-15,10e-15,num=100000, endpoint=True)
    sample_rate = 1/(time_vector[1]-time_vector[0])

    
    pulse1 = Pulse_Shaper_Dial(position, width, hole_position, hole_width, hole_depth, delay, sec_order, third_order, fourth_order)
    omega0 = 2356194490192345.0
    
    #E_field = 1*np.exp(1j*omega0*time_vector)*(1/(2*np.pi))* np.sqrt(np.pi/(-()))
    
    E_field = np.exp(-1.385*(time_vector/10e-15)**2)
    #plt.plot(time_vector, E_field)
    #plt.show()
    E_field_output,time_vector, E_field_output_ft, freq_vector, components_dict=pulse1.shape_input_pulse(E_field, time_vector, sample_rate, 
                                                                                                         generate_plots=True)
    
    #fwhm_list = [10e-15, 20e-15, 30e-15]
    fwhm_list = [10e-15]
    for fwhm in fwhm_list:
        lam0 = 800e-9
        w0 = 2*np.pi*c/(lam0)
        position = lam0
        pulse_dur = 10e-15
        delta_freq = 0.44/pulse_dur
        delta_lam = (delta_freq*lam0**2)/c
        width = delta_lam
        pulse2 = Pulse_Shaper_Dial(position, width, hole_position, hole_width, hole_depth, delay, sec_order, third_order, fourth_order)
        coefs = []
        phase = lambda time_vector: phase_func_gen(coefs, time_vector)
        inputPulse = produce_input_pulse_E_field(1, 10e-15, time_vector, pulse_type='gaussian', phase_func = phase)
        inputPulse = inputPulse*np.exp(1j*w0*time_vector)
        pulse2.shape_input_pulse(inputPulse, time_vector, sample_rate, generate_plots=True)
    #plt.plot(time_vector, E_field_output)
 
def produce_input_pulse_E_field(amplitude, fwhm, time_vector, pulse_type = 'gaussian', phase_func = None):
    if (pulse_type == 'gaussian'):
        E_field = amplitude*np.exp(-2*np.log(2)*(time_vector/fwhm)**2)
        E_field.astype(complex)
    
    if (phase_func):
        E_field = E_field*(phase_func(time_vector)).astype(complex)
        
    return E_field

def phase_func_gen(coefs, time_vector):
    phase_out = np.ones(len(time_vector), dtype=complex)
    expn = 0
    for coef in coefs:
        phase_out *= (np.exp(1j*coef*time_vector**expn)).astype(complex)
        expn += 1
    return phase_out
    
if __name__ == "__main__":
    main()
    
        
                 
    
    
    