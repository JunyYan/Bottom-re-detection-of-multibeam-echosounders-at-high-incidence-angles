from EMAllParser import *
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# from dataset import resize_sample
from WCP import *


matplotlib.rc('font', family='Palatino Linotype', size=10)


def predict_beam(beam_entry, model_1d, use_fft=False, fft_w=40):
    beam_angle = beam_entry['beam_angle']  # Beam pointing angle ref vertical in 0.01° 2S
    beam_data = beam_entry
    srs_no = beam_entry['srs_no']  # Start Range sample number, 0 to 65534
    Ns = beam_entry['Ns']  # Number of samples (Ns) 0 to 65534
    Dr = beam_entry['DR']
    amps = np.array(beam_entry['sam_amps']).reshape((Ns, 1)) * 0.5
    amps = resize_sample(amps[1:], 512, use_fft, fft_w)
    input = torch.tensor(amps)
    input = input.unsqueeze(0).unsqueeze(0)
    # output = model_1d(input.to(device, dtype=torch.float), input.to(device, dtype=torch.float))
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    output = model_1d(input.to(device, dtype=torch.float))
    output = output.squeeze(0).squeeze(0).cpu().detach().numpy()
    pre_bottom_id = np.argmax(output)
    return round(pre_bottom_id * Ns / 512)


def along_track_run(em_file, incidentAngel, model_file, use_fft = True, fft_w = 40):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model_1d = torch.load(model_file, map_location=torch.device(f'{device}'))

    em_parser = EMAllParser(em_file)
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

    ping_num = em_parser.datagram_count(0x6B)
    # ping_id = 100

    pre_bottom_ids = []
    det_bottom_ids = []

    for ping_id in range(ping_num):
        wc_dg = water_column_dgs[ping_id]

        for beam_entry in wc_dg['b_entries']:
            beam_angle = beam_entry['beam_angle']  # Beam pointing angle ref vertical in 0.01° 2S
            if abs(beam_angle * 0.01 - incidentAngel) < 0.5:
                beam_data = beam_entry
                srs_no = beam_entry['srs_no']  # Start Range sample number, 0 to 65534
                Ns = beam_entry['Ns']  # Number of samples (Ns) 0 to 65534
                Dr = beam_entry['DR']
                amps = np.array(beam_entry['sam_amps']).reshape((Ns, 1)) * 0.5
                amps = resize_sample(amps[1:], 512, use_fft, fft_w)
                input = torch.tensor(amps)
                input = input.unsqueeze(0).unsqueeze(0)


                # output = model_1d(input.to(device, dtype=torch.float), input.to(device, dtype=torch.float))
                output = model_1d(input.to(device, dtype=torch.float))
                output = output.squeeze(0).squeeze(0).cpu().detach().numpy()


                #             pre_bottom_id = np.argmax(output[300:430]) + 300
                pre_bottom_id = np.argmax(output)
                pre_bottom_ids.append(round(pre_bottom_id * Ns / 512))
                det_bottom_ids.append(Dr)
    #             if 100 < ping_id < 175:
    #                 plt.figure()
    #                 plt.plot(amps)
    #                 plt.plot(output)
                break # 不能少

    return det_bottom_ids, pre_bottom_ids
    # plt.figure()
    # plt.plot(det_bottom_ids)
    # plt.plot(pre_bottom_ids)
    # # plt.ylim(0, 1000)
    # plt.show()


def across_track_run(em_file, ping_id, model_file, use_fft = True, fft_w = 40):
    em_parser = EMAllParser(em_file)
    img_scale = 0.1  # image scale
    # ping_id = 180  # 371

    dg_n = em_parser.datagram_count(0x6B)
    dg_n = len(em_parser.wc_dg_start_nd)
    water_column_dgs = []
    # nds = []
    # max_row = 0
    for dgi in range(dg_n):
        # wc_dg = em_parser.parse_water_column_datagram(dgi)
        wc_dg = em_parser.parse_water_column_datagram_multi_datagram(dgi)
        # nds.append(wc_dg['Nd'])
        # if wc_dg['Nrx'] > max_row:
        #     max_row = wc_dg['Nrx']
        water_column_dgs.append(wc_dg)

    wc_dg = water_column_dgs[ping_id]
    net_model1 = torch.load(model_file, map_location=torch.device("cuda"))

    DRs_r = []
    DRs_p = []
    for beam_entry in wc_dg['b_entries']:
        beam_angle = beam_entry['beam_angle'] * 0.01  # Beam pointing angle ref vertical in 0.01° 2S
        srs_no = beam_entry['srs_no']  # Start Range sample number, 0 to 65534
        Ns = beam_entry['Ns']  # Number of samples (Ns) 0 to 65534
        DR = beam_entry['DR']
        pre_DR = predict_beam(beam_entry, net_model1, use_fft, fft_w)
        DRs_p.append(pre_DR)
        DRs_r.append(DR)

    return DRs_r, DRs_p