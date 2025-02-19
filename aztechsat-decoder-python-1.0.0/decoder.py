import struct
import json

significado={
    "ax100_hk":{
        "temp_pa": "Power amplifier temperature [tenth of °C]",
        "last_rssi":"Last RSSI",
        "bgnd_rssi":"GND RSSI"
    },
    "p31u_hk":{
        "vboost":"Power converters voltage [mV] [PV1, PV2, PV3]",
        "vbatt":"Battery voltage [mV]",
        "curin":"Current in [mA]",
        "cursun":"Boost converters current [mA]",
        "cursys":"Battery's output current [mA]",
        "curout":"Output current [mA]",
        "output":"Output channels status [0 or 1]",
        "wdt_i2c_time_left":"I2C wdt time left [s]",
        "wdt_gnd_time_left":"GND wdt time left [s]",
        "counter_wdt_gnd":"WDT GND reboot number",
        "temp":"Temperature sensor"

    },
    "a3200_hk":{
        "temp_a":"Temperature Sensor °C value",
        "cur_pwm":"PWM current",
        "boot_cause":"Last OBC reboot cause",
        "magneto_x":"Magnetometer - X axis",
        "magneto_y":"Magnetometer - Y axis",
        "magneto_z":"Magnetometer - Z axis",
        "gyro_x":"Gyro - X axis",
        "gyro_y":"Gyro - Y axis",
        "gyro_z":"Gyro - Z axis"
    },
    "pyl_hk":{
        "channel":"# of Channel",
        "no_of_bursts":"# of Bursts",
        "min_interval":"Minimum Burst Interval: Units of 5 seconds. Valid values are: 0x01 thru 0x3C (5 to 300 seconds)",
        "max_interval":"Maximum Burst Interval: Units of 5 seconds. Valid values are: 0x02 thru 0x78 (10 to 600 seconds)",
        "status":{
            "GpsSubsystemActive" :"GPS subsystem active",
            "ActiveMode" :"Device is in an active mode",
            "Reserved1" :"RESERVED DO NOT USE",
            "GpsFix" :"GPS Subsystem does have a fix",
            "GpsSubsystemFailure" :"GPS Subsystem failure",
            "SimplexTransmitterFailure" :"Simplex Transmitter failure",
            "BatteryLow" :"Battery low",
            "Reserved2" :"RESERVED DO NOT USE"
        },
        "seconds_since_last_transmission":"Number of seconds since the Device unit last attempted to send a satellite transmission.",
        "seconds_until_next_transmission":"Number of seconds until the Device unit attempts to send a satellite transmission.",
        "packet_size":"Packet size of last or current message.",
        "burst_number":"Currently waiting on or sending burst number",
        "seconds_until_burst_2":"Number of seconds until burst transmission number 2",
        "seconds_until_burst_3":"Number of seconds until burst transmission number 3",
        "total_tx_messages_current_mode":"Total messages transmitted in current mode.",
        "total_tx_packet_since_power_on":"Total Packet transmission count since hard power on.",
        "stingr_antenna_temp":"Stingr Antenna temperature",
        "lm70_temp":"Aguila Board temperature"
    },
    "adcs_hk":{
        "css":"Coarse Sun Sensors Value [order is +Y, +X, -X, -Y, -Z]",
        "panel_y_pos_temp":"Solar panel's temperature +Y",
        "panel_x_pos_temp":"Solar panel's temperature +X",
        "panel_x_neg_temp":"Solar panel's temperature -X",
        "panel_y_neg_temp":"Solar panel's temperature -Y",
        "panel_z_neg_temp":"Solar panel's temperature -Z",
        "status_bdot":"Bdot status [1:tumble2detumble, 0:valid, -1:no sample, -2:no previous sample -3:detumble2tumble]",
        "bdot_rate_slow":"Bdot value from low-pass filter slow",
        "bdot_rate_slow_2":"Bdot value from low-pass filter slow2",
        "bdot_detumb":"Value of detumbled state [0 for \"not detumbled\", 1 for \"detumbled\"]"
    },
    "metadata":{
        "clock_from_satellite":"Clock in unix format from the satelite"
    }
}

class Decoder():
    def __init__(self,data):
        self.data = data
    
    def to_int(self,data):
        return int.from_bytes(data,byteorder="big", signed=True)

    def to_uint(self,data):
        return int.from_bytes(data,byteorder="big", signed=False)

    def to_float(self,data):
        return struct.unpack('>f', data)[0]
    
    def dumpOBCdata(self):
        return self.dump_by_key('a3200_hk')
    
    def dumpEPSdata(self):
        return self.dump_by_key('p31u_hk')
    
    def dumpADCSdata(self):
        return self.dump_by_key('adcs_hk')
    
    def dumpRADIOdata(self):
        return self.dump_by_key('ax100_hk')
    
    def dumpPAYLOADdata(self):
        return self.dump_by_key('pyl_hk')
    
    def get_frame_id(self):
        decoded = json.loads(self.decode_to_json())
        return decoded['clock_from_satellite']
    
    def dump_by_key(self,key):
        decoded = json.loads(self.decode_to_json())
        dumped = []
        for data in decoded[key]:
            dumped.append([data,str(decoded[key][data]),significado[key][data]])
        #print(dumped)
        return dumped

    def decode_to_json(self):
        frame = self.data
        data={
        'ax100_hk':{
            'temp_pa':self.to_int(frame[44:46])/10.0 ,								#int16 Power amplifier temperature [tenth of °C]
            'last_rssi':self.to_int(frame[46:48]) ,								#int16 Last RSSI
            'bgnd_rssi':self.to_int(frame[48:50]) 								#int16 GND RSSI
        },
        'p31u_hk':{
            'vboost':[self.to_uint(frame[0:2]),self.to_uint(frame[2:4]),self.to_uint(frame[4:6])],#uint1    6 Power converters voltage [mV] [PV1, PV2, PV3]
            'vbatt':self.to_uint(frame[6:8]),									#uint16 Battery voltage [mV]
            'curin':[self.to_uint(frame[8:10]),self.to_uint(frame[10:12]),self.to_uint(frame[12:14])],								#uint16 Current in [mA]
            'cursun':self.to_uint(frame[14:16]),								#uint16 Boost converters current [mA]
            'cursys':self.to_uint(frame[16:18]),								#uint16 Battery's output current [mA]
            'curout':[self.to_uint(frame[18:20]),self.to_uint(frame[20:22]),self.to_uint(frame[22:24]),self.to_uint(frame[24:26])], #uint16 Output current [mA]
            'output':[frame[26],frame[27],frame[28],frame[29]],	#uint8 Output channels status [0 or 1]
            'wdt_i2c_time_left':self.to_uint(frame[30:34]),						#uint32 I2C wdt time left [s]
            'wdt_gnd_time_left':self.to_uint(frame[34:38]),						#uint32 GND wdt time left [s]
            'counter_wdt_gnd':self.to_uint(frame[38:42]),						#uint32 WDT GND reboot number
            'temp':self.to_int(frame[42:44]) 								#int16 Temperature sensors [0 = TEMP1, TEMP2, TEMP3, TEMP4, BATT0, BATT1]

        },
        'a3200_hk':{
            'temp_a':self.to_int(frame[50:52])/10.0,									#int16 Sensor A value
            'cur_pwm':self.to_uint(frame[52:54]),								#uint16 PWM current
            'boot_cause':self.to_uint(frame[54:58]),							#uint32 Last OBC reboot cause
            'magneto_x':self.to_float(frame[58:62]),								#float Magnetometer - X axis
            'magneto_y':self.to_float(frame[62:66]),								#float Magnetometer - Y axis
            'magneto_z':self.to_float(frame[66:70]),								#float Magnetometer - Z axis
            'gyro_x':self.to_float(frame[70:74]),									#float Gyro - X axis
            'gyro_y':self.to_float(frame[74:78]),									#float Gyro - Y axis
            'gyro_z':self.to_float(frame[78:82])									#float Gyro - Z axis
        },
        'pyl_hk':{
            'channel':frame[82],									#uint8 # of Channel
            'no_of_bursts':frame[83],								#uint8 # of Bursts
            'min_interval':frame[84],								#uint8 Minimum Burst Interval: Units of 5 seconds. Valid values are: 0x01 thru 0x3C (5 to 300 seconds)
            'max_interval':frame[85],								#uint8 Maximum Burst Interval: Units of 5 seconds. Valid values are: 0x02 thru 0x78 (10 to 600 seconds)
            'status':{
                'GpsSubsystemActive' :(frame[86]&1),           #bit GPS subsystem active
                'ActiveMode' :((frame[86])&2)>>1,					#bit Device is in an active mode
                'Reserved1' :((frame[86])&4)>>2,					#bit RESERVED DO NOT USE
                'GpsFix' :((frame[86])&8)>>3,						#bit GPS Subsystem does have a fix
                'GpsSubsystemFailure' :((frame[86])&16)>>4,			#bit GPS Subsystem failure
                'SimplexTransmitterFailure' :((frame[86])&32)>>5,    #bit Simplex Transmitter failure
                'BatteryLow' :((frame[86])&64)>>6,					#bit Battery low
                'Reserved2' :((frame[86])&128)>>7,					#bit RESERVED DO NOT USE
            },
            'seconds_since_last_transmission':self.to_uint(frame[87:89]),           #uint16 Number of seconds since the Device unit last attempted to send a satellite transmission.
            'seconds_until_next_transmission':self.to_uint(frame[89:91]),			#uint16 Number of seconds until the Device unit attempts to send a satellite transmission.
            'packet_size':frame[91],								#uint8 Packet size of last or current message.
            'burst_number':frame[92],								#uint8 Currently waiting on or sending burst number
            'seconds_until_burst_2':self.to_uint(frame[93:95]),						#uint16 Number of seconds until burst transmission number 2
            'seconds_until_burst_3':self.to_uint(frame[95:97]),						#uint16 Number of seconds until burst transmission number 3
            'total_tx_messages_current_mode':self.to_uint(frame[97:99]),			#uint16 Total messages transmitted in current mode.
            'total_tx_packet_since_power_on':self.to_uint(frame[99:101]),			#uint16 Total Packet transmission count since hard power on.
            'stingr_antenna_temp':frame[101],						#uint8 Stingr Antenna temperature
            'lm70_temp':self.to_uint(frame[102:104])/10.0									#uint16 Aguila Board temperature
        },
        'adcs_hk':{
            'css':[self.to_uint(frame[108:110]),
                self.to_uint(frame[110:112]),
                self.to_uint(frame[112:114]),
                self.to_uint(frame[114:116]),
                self.to_uint(frame[116:118])],								#uint16 Coarse Sun Sensors Value [order is +Y, +X, -X, -Y, -Z]
            'panel_y_pos_temp':self.to_float(frame[118:122]),							#float Solar panel's temperature +Y
            'panel_x_pos_temp':self.to_float(frame[122:126]),							#float Solar panel's temperature +X
            'panel_x_neg_temp':self.to_float(frame[126:130]),							#float Solar panel's temperature -X
            'panel_y_neg_temp':self.to_float(frame[130:134]),							#float Solar panel's temperature -Y
            'panel_z_neg_temp':self.to_float(frame[134:138]),							#float Solar panel's temperature -Z
            'status_bdot':frame[138],								#int8 Bdot status [1:tumble2detumble, 0:valid, -1:no sample, -2:no previous sample -3:detumble2tumble]
            'bdot_rate_slow':self.to_float(frame[139:143]),							#float Bdot value from low-pass filter slow
            'bdot_rate_slow_2':self.to_float(frame[143:147]),							#float Bdot value from low-pass filter slow2
            'bdot_detumb':frame[147]							#uint8 Value of detumbled state [0 for "not detumbled", 1 for "detumbled"]
        },
        'clock_from_satellite':(frame[104]<<24)+(frame[105]<<16)+(frame[106]<<8)+(frame[108])
    }
        data_json = json.dumps(data)
        return data_json
    
    def OBC_data(self):
        return "true"

def read_file(ruta):
    file = open(ruta, "rb")
    byte = file.read()
    if b'<HK>' in byte and b'</HK>' in byte:
        inicio = (byte.find(b'<HK>'))
        fin = (byte.find(b'</HK>'))
        return byte[inicio+4:fin]
    else:
        print("Frame Invalido")
        return "NO DATA"

def read_file_csv(ruta):
    file = open(ruta, "r")
    byte = file.read()
    for linea_str in byte.split('\n'):
        try:
            date=linea_str.split('|')[0]
            linea = bytes.fromhex(linea_str.split('|')[1])
            if b'<HK>' in linea and b'</HK>' in linea:
                inicio = (linea.find(b'<HK>'))
                fin = (linea.find(b'</HK>'))
                frame=linea[inicio+4:fin]
                return frame
            else:
                print("Frame Invalido")
                return "NO DATA"
        except:
            print("Ocurrio un error")
            return "NO DATA"

def extract_frame(data):
    try:
        linea = bytes.fromhex(data)
        if b'<HK>' in linea and b'</HK>' in linea:
            inicio = (linea.find(b'<HK>'))
            fin = (linea.find(b'</HK>'))
            frame=linea[inicio+4:fin]
            return frame
        else:
            print("Frame Invalido")
            return "NO DATA"
    except:
        print("Ocurrio un error")
        return "NO DATA"

