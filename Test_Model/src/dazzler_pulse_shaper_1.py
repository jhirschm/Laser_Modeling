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
    
    
    def __init__(self, position, width, hole_position, hole_width, hole_depth, delay, sec_order, third_order, fourth_order):
        self.position = position
        self.width = width
        self.hole_position = hole_position
        self.hole_depth = hole_depth
        self.hole_width = hole_width
        
        self.delay = delay
        self.sec_order = sec_order
        self.third_order = third_order
        self.fourth_order = fourth_order
        
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
        
        intensity_input = (np.abs(E_field_input))**2
        phase_input = -1*np.arctan(np.imag(E_field_input)/np.real(E_field_input))
        spectrum_freq_input = (np.abs(E_field_input_ft))**2
        #spectrum_wavelength_input = (2*np.pi*c/(wave_vec**2)) * (np.abs(E_field_input_wavelength_ft))**2
        spectral_phase_freq_input = -1*np.arctan(np.imag(E_field_input_ft)/np.real(E_field_input_ft))
        #spectra_phase_wavelength_input = -1*np.arctan(np.imag(E_field_input_wavelength_ft)/np.real(E_field_input_wavelength_ft))


        intensity_output = (np.abs(E_field_output))**2
        phase_output = -1*np.arctan(np.imag(E_field_output)/np.real(E_field_output))
        spectrum_freq_output = (np.abs(E_field_output_ft))**2
        #spectrum_wavelength_input = (2*np.pi*c/(wave_vec**2)) * (np.abs(E_field_output_wavelength_ft))**2
        spectral_phase_freq_input = -1*np.arctan(np.imag(E_field_output_ft)/np.real(E_field_output_ft))
        #spectra_phase_wavelength_input = -1*np.arctan(np.imag(E_field_output_wavelength_ft)/np.real(E_field_output_wavelength_ft))
 
        S_full_time = np.fft.ifft(S_full)
        intensity_transfer = (np.abs(S_full_time))**2
        phase_transfer= -1*np.arctan(np.imag(S_full_time)/np.real(S_full_time))
        spectrum_freq_transfer = (np.abs(S_full))**2
        #spectrum_wavelength_transfer = (2*np.pi*c/(wave_vec**2)) * (np.abs(E_field_input_wavelength_ft))**2
        spectral_phase_freq_transfer = -1*np.arctan(np.imag(S_full)/np.real(S_full))
        #spectra_phase_wavelength_trasnfer = -1*np.arctan(np.imag(E_field_input_wavelength_ft)/np.real(E_field_input_wavelength_ft))

        
        if (generate_plots):
            plot_list = [E_field_input, E_field_output, E_field_input_ft, E_field_output_ft]
            plot_list_labels = ["E Field Input", "E Field Output"]
            titles = ["Intensity", "Phase","Intensity", "Phase","Intensity", "Phase"]
            plot_list_domains = [time_vector, freq_vector]
            
            self.standard_plot_pattern(time_vector, intensity_input,phase_input,
                                  intensity_transfer, phase_transfer, intensity_output,phase_output,
                                  "Input, Transfer, Output", titles, "time (ps)", "arb", x_scale = 1e-15)
            self.standard_plot_pattern(freq_vector, spectrum_freq_input,spectral_phase_freq_input,
                                  spectrum_freq_transfer, spectral_phase_freq_transfer, spectrum_freq_output,spectral_phase_freq_input,
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
                              x_scale = 1):
        xvec = xvec/x_scale
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
        
            
        
        plt.show()
        
        return
        
        
        
        
    
    
def main():
    position = 800e-9
    width = 400e-9#93.9e-9
    hole_position = 700e-9
    hole_width = 300e-9#3e-9
    hole_depth=1.5
    delay=1e-14#4250e-15
    sec_order=0#22000*(1e-15)**2
    third_order=0*(1e-15)**3
    fourth_order=0*(1e-15)**4
    pulse1 = Pulse_Shaper_Dial(position, width, hole_position, hole_width, hole_depth, delay, sec_order, third_order, fourth_order)
    omega0 = 2356194490192345.0
    #E_field = 1*np.exp(1j*omega0*time_vector)*(1/(2*np.pi))* np.sqrt(np.pi/(-()))
    time_vector = np.linspace(-20e-15,20e-15,num=10000, endpoint=True)
    sample_rate = 1/(time_vector[1]-time_vector[0])
    E_field = np.exp(-1.385*(time_vector/10e-15)**2)
    #plt.plot(time_vector, E_field)
    #plt.show()
    E_field_output,time_vector, E_field_output_ft, freq_vector, components_dict=pulse1.shape_input_pulse(E_field, time_vector, sample_rate, 
                                                                                                         generate_plots=True)
    #plt.plot(time_vector, E_field_output)
    
    
if __name__ == "__main__":
    main()
    
        
                 
    
    
    