import struct
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

EM3_START_BYTE = 0x02
EM3_END_BYTE = 0x03
EM3_END = 0x03

G_EM_TYRES_STR = {
    # Multibeam data
    0x44: 'EM_Depth_Datagram              ',  # 68  K This datagram is used for EM 2000, 3000, 3002, 1002, 300 & 120.
    0x58: 'EM_XYZ_88                      ',  # 88  X This datagram replaces the previous depth (D) datagram for
                                              # the new multibeam models (EM 2040, EM 710, EM 122, EM 302, ME 70).
    0x4B: 'EM_Central_Beams_Echogram      ',  # 75  K This datagram is only available for EM 120 and EM 300.
    0x46: 'EM_Raw_Range_and_Beam_Angle_F  ',  # 70  F Only used for EM 3000, old
    0x66: 'EM_Raw_Range_and_Beam_Angle_f  ',  # 102 f Used for EM 120, EM 300, EM 1002, EM 2000, EM 3000 and EM 3002
    0x4e: 'EM_Raw_Range_and_Angle_78      ',  # 78  N This datagram replaces the previous Raw range and beam angle f
                                              # datagram for the new multibeam models (EM 2040, 710, 302, 122, 70).
    0x53: 'EM_Seabed_image_datagram       ',  # 83  S This datagram is used for EM 2000, 3000, 3002, 1002, 300 and 120.
    0x59: 'EM_Seabed_Image_Data_89        ',  # 89  Y This datagram replaces the previous Seabead image (S) datagram
                                              # for the new multibeam models (EM 2040, EM 710, EM 302, EM 122, ME 70).
    0x6B: 'EM_Water_Column_Datagram       ',  # 107 k Used for EM 122, EM 302, EM 710, EM 2040, EM 3002 and ME 70.
    # Extenal sensors
    0x41: 'EM_Attitude_Datagram           ',  # 65  A(ttitude data)
    0x6E: 'EM_Net_Att_Velocity_datagram   ',  # 110 n(etwork data)
    0x43: 'EM_Clock_Datagram              ',  # 67  C(lock data)
    0x68: 'EM_Depth_or_Height_Datagram    ',  # 104 h(eight data)
    0x48: 'EM_Heading_Datagram            ',  # 72  H(eading data)
    0x50: 'EM_Position_Datagrams          ',  # 80  P(osition data)
    0x45: 'EM_Single_beam_depth_Datagram  ',  # 69  E(cho sounder data)
    0x54: 'EM_Tide_Datagram               ',  # 84  T(ide data)
    # Sound speed
    0x47: 'EM_Surface_Sound_Speed         ',  # 71  G
    0x55: 'EM_Sound_Speed_Profile_Datagram',  # 85  U This datagram will contain the profile actually used
                                              # in the real time ray-bending calculations to convert range and angle
                                              # to xyz data. It will usually be issued together with
                                              # the installation parameter datagram.
    0x56: 'EM_Sound_Velocity_Profile      ',  # 86  V A little different from 0x55, used by older EM3000
    0x57: 'EM_Kongsberg_Maritime_SSP      ',  # 87  W
    # Multibeam parameters
    0x49: 'EM_Install_Parameters_Start    ',  # 73  I(nstallation parameters)
    0x69: 'EM_Install_Parameters_Stop     ',  # 105 i(nstallation parameters)
    0x70: 'EM_Install_Parameters_Remote   ',  # !!! error exist!!! 112 p !=114 r(emote information) 0x72???
    0x52: 'EM_Runtime_Parameters          ',  # 82  R(untime parameter)
    0x4A: 'EM_Mechanical_Transducer_Tilt  ',  # 74  J
    0x33: 'EM_ExtraParameters_Datagram    ',  # 51  3
    # PU information and status
    0x30: 'EM_PU_Id_output                ',  # 48  0
    0x31: 'EM_PU_Status_output            ',  # 49  1
    0x42: 'EM_PU_BIST_result_output       '   # 66  B
}


class EMAllParser:

    def __init__(self, file_path):
        self.file_path = file_path
        self.datagrams = []
        self.datagram_sizes = []
        self.datagrams_positions = []
        self.en = '<'                                   # big endian '>' or little endian '<'
        self.fh = open(file_path, 'rb')
        self.pre_parse()
        # self.datagrams = np.array(self.datagrams)       # 按照记录顺序保存所有包
        # self.u_datagrams = np.unique(self.datagrams)    # 包类型唯一化之后

    # 预解码
    def pre_parse(self):
        position = 0
        self.detect_endian_type()
        while True:
            packet_head = self.parse_datagram_header()
            if len(packet_head) == 0:
                break
            if packet_head['start_id'] != 0x02:
                print('Start identifier fail!')
                break
            self.datagrams_positions.append(position)
            self.datagrams.append(packet_head['type_of_datagram'])
            self.datagram_sizes.append(packet_head['number_of_bytes'])
            position = position + packet_head['number_of_bytes'] + 4
            self.fh.seek(packet_head['number_of_bytes'] - 20 + 4, 1)

        self.datagrams = np.array(self.datagrams)       # 按照记录顺序保存所有包
        self.u_datagrams = np.unique(self.datagrams)    # 包类型唯一化之后

        # 检测水柱包
        wc_dg_n = len(np.where(self.datagrams == 107)[0])
        # for dgi in range(wc_dg_n):
        self.wc_dg_start_nd = []
        dgi = 0
        while dgi < wc_dg_n:
            wc_dg = self.parse_water_column_datagram(dgi)
            self.wc_dg_start_nd.append([dgi, wc_dg['Nd']])
            dgi = dgi + wc_dg['Nd']
        # print(self.wc_dg_start_nd)
        # print(len(self.wc_dg_start_nd))
            # print(wc_dg['Nd'])
        # water_column_dgs = []
        # nds = []
        # max_row = 0
        # for dgi in range(wc_dg_n):
        #     wc_dg = self.parse_water_column_datagram(dgi)
        #     nds.append(wc_dg['Nd'])
        #     if wc_dg['Nrx'] > max_row:
        #         max_row = wc_dg['Nrx']
        #     water_column_dgs.append(wc_dg)


    # 解码包头
    def detect_endian_type(self):
        head_content = self.fh.read(20)
        if head_content:
            # packet_head_ = struct.unpack('<IBBHIIHH', head_content)
            packet_head_ = struct.unpack('>IBBHIIHH', head_content)
            em_model_number = packet_head_[3]
            date = packet_head_[4]

        if 0 < em_model_number < 4000 and 19000000 < date < 21000000:
            self.en = '>'
        self.fh.seek(0, 0)

    # 解码包头
    def parse_datagram_header(self):
        head_content = self.fh.read(20)
        if head_content:
            packet_head_ = struct.unpack(f'{self.en}IBBHIIHH', head_content)
            return dict(
                number_of_bytes=packet_head_[0],
                start_id=packet_head_[1],
                type_of_datagram=packet_head_[2],
                em_model_number=packet_head_[3],
                date=packet_head_[4],
                millsecond=packet_head_[5],
                sequential_counter=packet_head_[6],
                serial_number=packet_head_[7]
            )
        else:
            return dict()

    def datagram_count(self, dg):
        return len(np.where(self.datagrams == dg)[0])

    def get_datagram_position(self, dg_type, dg_id):
        indices = np.where(self.datagrams == dg_type)
        return self.datagrams_positions[indices[0][dg_id]]

    def list_datagram(self, type):
        return [self.parse_datagram(type, dgi) for dgi in range(self.datagram_count(type))]

    # 统一的解码接口，输入包类型与包id
    def parse_datagram(self, type, dg_id):
        parse_function = {
            0x44: self.parse_depth_datagram,
            # 0x58: 'EM_XYZ_88                      ',
            # 0x4B: 'EM_Central_Beams_Echogram      ',
            # 0x46: 'EM_Raw_Range_and_Beam_Angle_F  ',
            0x66: self.parse_raw_range_and_beam_angle_f,
            # 0x4e: 'EM_Raw_Range_and_Angle_78      ',
            0x53: self.parse_seabed_image_datagram,
            # 0x59: 'EM_Seabed_Image_Data_89        ',
            0x6B: self.parse_water_column_datagram,
            0x41: self.parse_attitude_datagram,
            # 0x6E: 'EM_Net_Att_Velocity_datagram   ',
            # 0x43: 'EM_Clock_Datagram              ',
            # 0x68: 'EM_Depth_or_Height_Datagram    ',
            # 0x48: 'EM_Heading_Datagram            ',
            0x50: self.parse_position_datagram,
            # 0x45: 'EM_Single_beam_depth_Datagram  ',
            # 0x54: 'EM_Tide_Datagram               ',
            # 0x47: 'EM_Surface_Sound_Speed         ',
            0x55: self.parse_sound_speed_profile_datagram,
            # 0x56: 'EM_Sound_Velocity_Profile      ',
            # 0x57: 'EM_Kongsberg_Maritime_SSP      ',
            # 0x49: 'EM_Install_Parameters_Start    ',
            # 0x69: 'EM_Install_Parameters_Stop     ',
            # 0x70: 'EM_Install_Parameters_Remote   ',
            # 0x52: 'EM_Runtime_Parameters          ',
            # 0x4A: 'EM_Mechanical_Transducer_Tilt  ',
            # 0x33: 'EM_ExtraParameters_Datagram    ',
            # 0x30: 'EM_PU_Id_output                ',
            0x31: self.parse_pu_status_output
            # 0x42: 'EM_PU_BIST_result_output       '
        }
        return parse_function[type](dg_id)

    # 0x31
    def parse_pu_status_output(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x31, dg_id), 0)

    # 0x41
    def parse_attitude_datagram(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x41, dg_id), 0)
        header = self.parse_datagram_header()
        N = struct.unpack(f'{self.en}H', self.fh.read(2))[0]
        atts = []
        for i in range(N):
            content = self.fh.read(12)
            att_data = struct.unpack(f'{self.en}2H3hH', content)
            atts.append({
                'time': att_data[0],
                'status': att_data[1],
                'roll': att_data[2] * 0.01,
                'pitch': att_data[3] * 0.01,
                'heave': att_data[4] * 0.01,
                'heading': att_data[5] * 0.01
            })
        return {
            'header': header,
            'atts': atts
        }

    # Clock datagrams 0x43
    def parse_clock_datagram(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x43, dg_id), 0)

    # 0x44
    def parse_depth_datagram(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x44, dg_id), 0)
        header = self.parse_datagram_header()

        content = self.fh.read(12)
        data = struct.unpack(f'{self.en}3H4BH', content)

        N = data[4]
        beam_depths = []
        for bi in range(N):
            content = self.fh.read(16)
            beam_data = struct.unpack(f'{self.en}H2h3H2BbB', content)
            beam_depths.append(dict(
                z=beam_data[0],         # (z) depth from transmit transducer (unsigned for EM 120 and EM 300)
                y=beam_data[1],         # (y) across-track distance -32768 to 32766
                x=beam_data[2],         # (x) along-track distance -32768 to 32766
                depa=beam_data[3],      # in 0.01° -11000 to 11000
                azia=beam_data[4],      # in 0.01° 0 to 56999
                range=beam_data[5],     # (one - way travel time) 0 to 65534
                QF=beam_data[6],        # 0 to 254
                det_wl=beam_data[7],    # length_of_detection_window (samples/4) 1 to 254
                BS=beam_data[8],        # reflectivity(BS) in 0.5 dB resolution) -128 to +126
                bN=beam_data[9]         # 1 to 254
            ))

        return {
            'header': header,
            'heading': data[0],                 # heading of vessel in 0.01°
            'sound_speed': data[1],             # sound speed at transducer in dm/s
            'transducer_depth_re': data[2],     # Transmit Transducer depth re water level at time of ping in cm
            'max_beam_num': data[3],            # maximum number of beams possible >48
            'N': data[4],                       # valid_beam_num N, 1-254
            'z_resolution': data[5],            # in cm 1-254
            'x_y_resolution': data[6],          # in cm 1-254
            'sampling_rate': data[7],           # f in Hz
                                                # or Depth difference between sonar heads in the EM 3000D
            'depth_entries': beam_depths
        }

    # 0x49
    def parse_install_parameters_start(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x49, dg_id), 0)

    # P(osition data) (Always 050h)
    def parse_position_datagram(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x50, dg_id), 0)
        header = self.parse_datagram_header()
        content = self.fh.read(18)
        pos_data = struct.unpack(f'{self.en}2i4H2B', content)
        return {
            'header': header,
            'lat': pos_data[0] / 20000000,  # in decimal degrees*20000000 (- if S) (E: 32°34’ S = -651333333)
            'lon': pos_data[1] / 10000000,  # in decimal degrees*10000000 (- if W) (E: 110.25° E = 1102500000 )
            'pos_fQ': pos_data[2] * .01,    # measure of position fix quality in cm
            'v_speed': pos_data[3] * .01,   # over ground in cm/s >0
            'v_course': pos_data[4] * .01,  # over ground in 0.01° 0 to 35999
            'v_heading': pos_data[5] * .01, # in 0.01° 0 to 35999
            'psd': pos_data[6],             # 1 to 254
            'nbi': pos_data[7]              # <=254
        }

    # 0x52
    def parse_runtime_parameters(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x52, dg_id), 0)

    # 0x53
    def parse_seabed_image_datagram(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x53, dg_id), 0)
        header = self.parse_datagram_header()

        content = self.fh.read(16)
        img_data = struct.unpack(f'{self.en}5H2bH2B', content)

        N = img_data[9]
        beam_infos = []
        for bi in range(N):
            content = self.fh.read(6)
            beam_data = struct.unpack(f'{self.en}BbHH', content)
            beam_infos.append(dict(
                beam_index_number=beam_data[0],  # 1U 0 to 253 The beam index number is the beam number - 1.
                sorting_direction=beam_data[1],  # -1/1 The 1st sample in a beam has lowest range if 1, highest if -1.
                Ns=beam_data[2],                 # samples_num per beam = Ns
                centre_sample_num=beam_data[3]   # The centre sample number is the detection point of a beam.
            ))

        ping_samples = []
        for bi in range(N):
            Ns = beam_infos[bi]['Ns']
            samples = struct.unpack(f'{self.en}{Ns}b', self.fh.read(Ns))
            ping_samples.append(samples)

        return dict(
            header=header,
            m_ab_co=img_data[0],            # mean_absorption_coefficient in 0.01 dB/km  1 to 20000
            pulse_length=img_data[1],       # in μs >50
            RI=img_data[2],                 # to normal incidence used to correct sample amplitudes in no. of samples
            start_rs=img_data[3],           # Start range sample of TVG ramp if not enough, dynamic range (0 else)
            stop_rs=img_data[4],            # Stop range sample of TVG ramp if not enough, dynamic range (0 else)
            BSN=img_data[5],                # normal_incidence_BS in dB (BSN)  (Example: -20 dB = 236)
            BSO=img_data[6],                # oblique_BS in dB (BSO) (Example:–1 dB = 255)
            tx_beamwidth=img_data[7],       # in 0.1°
            Crossover_Angle=img_data[8],    # in 0.1°
            N=img_data[9],                  # valid_beams_num (N) 1 to 254
            beam_entries=beam_infos,
            sample_entries= ping_samples
        )

    # Sound speed profle datagram 0x55
    def parse_sound_speed_profile_datagram(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x55, dg_id), 0)
        header = self.parse_datagram_header()
        ssp_data = struct.unpack(f'{self.en}2I2H', self.fh.read(12))
        N = ssp_data[2]
        ssp = []
        for i in range(N):
            ssp_entries = struct.unpack(f'{self.en}2I', self.fh.read(8))
            ssp.append(dict(depth=ssp_entries[0],
                            speed=ssp_entries[1]*0.1))  # Sound speed in dm/s 14000 to 17000
        return {
            'header': header,
            'data': ssp_data[0],
            'time': ssp_data[1],
            'N': ssp_data[2],
            'resolution': ssp_data[3] * 0.01,   # Depth resolution in cm
            'ssp': ssp
        }

    # Raw range and beam angle datagrams 0x66
    def parse_raw_range_and_beam_angle_f(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x66, dg_id), 0)
        header = self.parse_datagram_header()
        rrba_data = struct.unpack(f'{self.en}2HIi4H', self.fh.read(20))
        Ntx = rrba_data[0]
        N =  rrba_data[1]
        tx_enties = []
        for i in range(Ntx):
            tx_data = struct.unpack(f'{self.en}hH3IH2B', self.fh.read(20))
            tx_enties.append({
                'TileAngle': tx_data[0],
                'FocusRange': tx_data[1],
                'SignalLength': tx_data[2],     # in us
                'TimeOffset': tx_data[3],       # in us
                'CenFrequency': tx_data[4],
                'Bandwidth': tx_data[5],
                'SWIdentifier': tx_data[6],
                'TxNo.': tx_data[7]
            })
        beam_entires = []
        for i in range(N):
            beam_data = struct.unpack(f'{self.en}hHBb2BhH', self.fh.read(12))
            beam_entires.append({
                'PointAngle': beam_data[0] * .01,
                'Range0.25': beam_data[1] / 4.,
                'TxNo.': beam_data[2],
                'BS': beam_data[3] * 0.5,
                'QF': beam_data[4],
                'DetectionWL': beam_data[5],
                'BeamNo': beam_data[6]
            })
        return {
            'header': header,
            'Ntx': Ntx,
            'N': N,
            'SampleF': rrba_data[2] * 0.01,
            'ROVdepth': rrba_data[3],
            'SoundSpeed': rrba_data[4] * 0.1,
            'MaxBeamNo.': rrba_data[5],
            'TxEntries': tx_enties,
            'BeamEntries': beam_entires
        }

    # 0x69
    def parse_install_parameters_stop(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x69, dg_id), 0)

    # 0x6B Water column datagram
    def parse_water_column_datagram(self, dg_id):
        self.fh.seek(self.get_datagram_position(0x6B, dg_id), 0)

        all_head_s = self.fh.read(20)
        packet_head_ = struct.unpack(f'{self.en}IBBHIIHH', all_head_s)

        content = self.fh.read(24)
        wc_datagram_ = struct.unpack(f'{self.en}HHHHHHIhBbB3B', content)

        em_wctx_entries = []
        for i in range(wc_datagram_[2]):  # 1 to 20 Number of transmit sectors
            content = self.fh.read(6)
            em_wctx_entry_ = struct.unpack(f'{self.en}hHBB', content)
            em_wctx_entry = {
                'tilt_angle': em_wctx_entry_[0],  # re TX array in 0.01°
                'center_frequency': em_wctx_entry_[1],  # in 10 Hz — 1000 to 50000
                'tx_no': em_wctx_entry_[2]  # transmit sector number — 0 to 19
            }
            em_wctx_entries.append(em_wctx_entry)

        em_wcb_entries = []
        for i in range(wc_datagram_[4]):  # Number of beams in this datagram = Nrx
            content = self.fh.read(10)
            em_wcb_entry_ = struct.unpack(f'{self.en}hHHHBB', content)
            nsample = em_wcb_entry_[2]
            content = self.fh.read(nsample)
            em_wcb_amps = struct.unpack(f'{self.en}{nsample}b', content)
            em_wcb_entry = {
                'beam_angle': em_wcb_entry_[0],  # Beam pointing angle ref vertical in 0.01° 2S
                'srs_no': em_wcb_entry_[1],  # Start Range sample number, 0 to 65534
                'Ns': em_wcb_entry_[2],  # Number of samples (Ns) 0 to 65534
                'DR': em_wcb_entry_[3],  # Detected range in samples (DR) 0 to 65534
                'tx_no': em_wcb_entry_[4],  # Transmit sector number, 0 to 19
                'beam_no': em_wcb_entry_[5],  # 0 to 254 1U Beam number
                'sam_amps': em_wcb_amps
            }
            em_wcb_entries.append(em_wcb_entry)

        return {
            'number_of_bytes': packet_head_[0],
            'start_id': packet_head_[1],
            'type_of_datagram': packet_head_[2],
            'em_model_number': packet_head_[3],
            'date': packet_head_[4],
            'millsecond': packet_head_[5],
            'sequential_counter': packet_head_[6],
            'serial_number': packet_head_[7],
            'Nd': wc_datagram_[0],  # 1 to Nd Number of datagrams
            'datagram_numbers': wc_datagram_[1],  # 1 to Nd Datagram numbers
            'Ntx': wc_datagram_[2],  # 1 to 20  Number of transmit sectors
            'total_no': wc_datagram_[3],  # 1 to Nd  Total no. of receive beams
            'Nrx': wc_datagram_[4],  # 1 to Nd  Number of beams in this datagram = Nrx
            'SS': wc_datagram_[5],  # 14000 to 16000 2U Sound speed in 0.1 m/s (SS)
            'SF': wc_datagram_[6],  # 1000 to 4000000 Sampling frequency in 0.01 Hz resolution (SF)
            'TX_time_heave': wc_datagram_[7],  # 1000 to 1000 2S TX time heave (at transducer) in cm
            'X': wc_datagram_[8],  # 20 to 40 1U TVG function applied (X)
            'C': wc_datagram_[9],  # TVG offset in dB (C)
            'scanning_info': wc_datagram_[10],  # Scanning info.
            'tx_entries': em_wctx_entries,
            'b_entries': em_wcb_entries
        }

    def parse_water_column_datagram_multi_datagram(self, dg_st_id):
        st_nd = self.wc_dg_start_nd[dg_st_id]
        # self.fh.seek(self.get_datagram_position(0x6B, st_nd[0]), 0)
        if st_nd[1] <= 0:
            return
        wc_dg_0 = self.parse_water_column_datagram(st_nd[0])
        # wc_dg_md = []
        for id in range(1, st_nd[1]):
            wc_dg = self.parse_water_column_datagram(st_nd[0] + id)
            # wc_dg_md.append(wc_dg)
            wc_dg_0['b_entries'].extend(wc_dg['b_entries']) # 将多个包数据进行组合
        # print(wc_dg_0)
        return wc_dg_0

    # Print
    def __str__(self):
        _str = ''
        for dg in self.u_datagrams:
            _str += f'Datagram Type {dg}\t: {G_EM_TYRES_STR[dg]}\n'
        _str = 'Datagram \t\t Type \t\t\t\t\t\tNumber \n'
        for dg in self.u_datagrams:
            _str += f'{dg}\t\t: {G_EM_TYRES_STR[dg]} '
            _n = np.where(self.datagrams == dg)[0].size
            _str += f'{_n}\n'
        return _str
