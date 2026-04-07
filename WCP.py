import matplotlib.pyplot as plt

from EMAllParser import *
import numpy as np
import cv2
from scipy import interpolate
from scipy.fft import fft, ifft


def one_beam(em_parser, ping_id, inc_angle):
    dg_n = em_parser.datagram_count(0x6B)
    water_column_dgs = []
    nds = []
    max_row = 0
    for dgi in range(dg_n):
        wc_dg = em_parser.parse_water_column_datagram(dgi)
        nds.append(wc_dg['Nd'])
        if wc_dg['Nrx'] > max_row:
            max_row = wc_dg['Nrx']
        water_column_dgs.append(wc_dg)

    ping_amps = []
    wc_dg = water_column_dgs[ping_id]
    SS = wc_dg['SS']
    SF = wc_dg['SF']

    min_amp = 128.
    max_amp = -128.
    max_width = 0
    max_depth = 0
    # plt.figure()
    for beam_entry in wc_dg['b_entries']:
        beam_angle = beam_entry['beam_angle']  # Beam pointing angle ref vertical in 0.01° 2S
        if abs(beam_angle * 0.01 - inc_angle) < 0.5:
            srs_no = beam_entry['srs_no']  # Start Range sample number, 0 to 65534
            Ns = beam_entry['Ns']  # Number of samples (Ns) 0 to 65534
            amps = np.array(beam_entry['sam_amps']).reshape((Ns, 1)) * 0.5
            # plt.plot(amps)
            # plt.vlines(beam_entry['DR'], -64, 0, colors='orange', linestyles='dashdot')
            DR = np.ones(Ns) * -64.
            if beam_entry['DR'] == 0:
                DR[beam_entry['DR'] + 1] = 0
            else:
                DR[beam_entry['DR']] = 0
            return (amps, DR)


def draw_one_beam(em_parser, data_id, inc_angle, plt):
    dg_n = em_parser.datagram_count(0x6B)
    water_column_dgs = []
    nds = []
    max_row = 0
    for dgi in range(dg_n):
        wc_dg = em_parser.parse_water_column_datagram(dgi)
        nds.append(wc_dg['Nd'])
        if wc_dg['Nrx'] > max_row:
            max_row = wc_dg['Nrx']
        water_column_dgs.append(wc_dg)

    ping_amps = []
    wc_dg = water_column_dgs[data_id]
    SS = wc_dg['SS']
    SF = wc_dg['SF']

    min_amp = 128.
    max_amp = -128.
    max_width = 0
    max_depth = 0
    # plt.figure()
    for beam_entry in wc_dg['b_entries']:
        beam_angle = beam_entry['beam_angle']  # Beam pointing angle ref vertical in 0.01° 2S
        if abs(beam_angle * 0.01 - inc_angle) < 1.0: # 相差角度范围 默认0.5 EM710改成1
            srs_no = beam_entry['srs_no']  # Start Range sample number, 0 to 65534
            Ns = beam_entry['Ns']  # Number of samples (Ns) 0 to 65534
            amps = np.array(beam_entry['sam_amps']).reshape((Ns, 1)) * 0.5
            plt.plot(amps)
            plt.vlines(beam_entry['DR'], -64, 0, colors='orange', linestyles='dashdot')
            break
    # plt.show()


def one_beam_data(wc_dg, inc_angle):
    ping_amps = []
    SS = wc_dg['SS']
    SF = wc_dg['SF']
    # plt.figure()
    for beam_entry in wc_dg['b_entries']:
        beam_angle = beam_entry['beam_angle']  # Beam pointing angle ref vertical in 0.01° 2S
        if abs(beam_angle * 0.01 - inc_angle) < 0.5:
            srs_no = beam_entry['srs_no']  # Start Range sample number, 0 to 65534
            Ns = beam_entry['Ns']  # Number of samples (Ns) 0 to 65534
            amps = np.array(beam_entry['sam_amps']).reshape((Ns, 1)) * 0.5
            return amps, beam_entry['DR']


def draw_one_beam2(wc_dg, inc_angle, plt):
    ping_amps = []
    SS = wc_dg['SS']
    SF = wc_dg['SF']
    # plt.figure()
    for beam_entry in wc_dg['b_entries']:
        beam_angle = beam_entry['beam_angle']  # Beam pointing angle ref vertical in 0.01° 2S
        if abs(beam_angle * 0.01 - inc_angle) < 0.5:
            srs_no = beam_entry['srs_no']  # Start Range sample number, 0 to 65534
            Ns = beam_entry['Ns']  # Number of samples (Ns) 0 to 65534
            amps = np.array(beam_entry['sam_amps']).reshape((Ns, 1)) * 0.5
            plt.plot(amps)
            plt.vlines(beam_entry['DR'], -64, 0, colors='orange', linestyles='dashdot')
            break
    # plt.show()


def wc_alongtrack_seabed(em_parser, inc_angle):
    dg_n = em_parser.datagram_count(0x6B)
    water_column_dgs = []
    nds = []
    max_row = 0
    for dgi in range(dg_n):
        wc_dg = em_parser.parse_water_column_datagram(dgi)
        nds.append(wc_dg['Nd'])
        if wc_dg['Nrx'] > max_row:
            max_row = wc_dg['Nrx']
        water_column_dgs.append(wc_dg)

    al_seabed = []
    for wc_dg in water_column_dgs:
        for beam_entry in wc_dg['b_entries']:
            beam_angle = beam_entry['beam_angle']  # Beam pointing angle ref vertical in 0.01° 2S
            if abs(beam_angle * 0.01 - inc_angle) < 1.0:
                DR = beam_entry['DR']
                al_seabed.append(DR)
                break
    return al_seabed


def wc_alongtrack_image(em_parser, inc_angle, showSeabed=False, newDrs=[]):
    dg_n = em_parser.datagram_count(0x6B)
    water_column_dgs = []
    nds = []
    max_row = 0
    for dgi in range(dg_n):
        wc_dg = em_parser.parse_water_column_datagram(dgi)
        nds.append(wc_dg['Nd'])
        if wc_dg['Nrx'] > max_row:
            max_row = wc_dg['Nrx']
        water_column_dgs.append(wc_dg)

    ping_amps = []
    DRs = []
    for wc_dg in water_column_dgs:
        SS = wc_dg['SS']
        SF = wc_dg['SF']

        min_amp = 128.
        max_amp = -128.
        max_width = 0
        max_depth = 0
        for beam_entry in wc_dg['b_entries']:
            beam_angle = beam_entry['beam_angle']  # Beam pointing angle ref vertical in 0.01° 2S
            if abs(beam_angle * 0.01 - inc_angle) < 0.5:
                srs_no = beam_entry['srs_no']  # Start Range sample number, 0 to 65534
                Ns = beam_entry['Ns']  # Number of samples (Ns) 0 to 65534
                amps = np.array(beam_entry['sam_amps']).reshape((Ns, 1)) * 0.5
                DR = beam_entry['DR']
                # slant_ranges = (np.arange(srs_no, srs_no + Ns) * (SS * 0.1) / ((SF * 0.01) * 2)).reshape((Ns, 1))
                # x = slant_ranges * math.sin(math.radians(beam_angle * 0.01))
                # d = slant_ranges * math.cos(math.radians(beam_angle * 0.01))
                # beam_amps = np.concatenate((x, d, amps), axis=1)
                ping_amps.append(amps)
                DRs.append(DR)
                min_amp = min_amp if min_amp < np.min(amps) else np.min(amps)
                max_amp = max_amp if max_amp > np.max(amps) else np.max(amps)
                break
                #

                # max_depth = max_depth if max_depth > np.max(d) else np.max(d)
                # max_width = max_width if max_width > np.max(np.abs(x)) else np.max(np.abs(x))
    if len(newDrs) > 0:
        DRs = newDrs

    max_height = 0
    for beams in ping_amps:
        max_height = max_height if max_height > len(beams) else len(beams)
        # img_width = int(max_width / img_scale) + 1
        # img_height = int(max_depth / img_scale) + 1
    min_amp = -64
    max_amp = -10
    wc_img = np.ones((max_height, len(ping_amps)), dtype=np.uint8) * 0  # -64
    for c in range(len(ping_amps)):
        wc_img[0:len(ping_amps[c]), c] = np.squeeze(((ping_amps[c] - min_amp)
                                                     / (max_amp - min_amp) * 255).astype(np.uint8))

    if showSeabed:
        wc_img = cv2.cvtColor(wc_img, cv2.COLOR_GRAY2RGB)  # wc_img 0-255?
        for c in range(len(ping_amps) - 1):
            cv2.line(wc_img, (c, round(DRs[c])), (c + 1, round(DRs[c + 1])), (255, 0, 0), 2)
            # wc_img[DRs[c], c] = (255, 0, 0)

    return wc_img



# 按长度重采样
def resize_sample(seq, echo_length, use_fft = True, w = 40):
    x = np.arange(len(seq))
    tck = interpolate.splrep(x, seq, s=0)
    xn = np.arange(x[0], x[-1], (x[-1] - x[0]) / echo_length)
    yn = interpolate.splev(xn, tck, der=0)

    l = int(w / 2) # default 20
    if use_fft:
        # plt.figure()
        # plt.plot(yn)
        yf = fft(yn)
        yf[l:-l] = 0
        yn = ifft(yf)
        # plt.plot(yn)
        # plt.show()

    if len(yn) > echo_length:
        yn = yn[0:echo_length]
    yn = (yn - min(yn)) / (max(yn) - min(yn))
    # yn = yn / (2**16)
    return yn
