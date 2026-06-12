# config.py
RAW_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Maps every specific attack name to one of 5 broad categories
ATTACK_MAPPING = {
    # DoS — Denial of Service
    'neptune': 'DoS', 'smurf': 'DoS', 'back': 'DoS', 'teardrop': 'DoS',
    'pod': 'DoS', 'land': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS', 'worm': 'DoS',
    # Probe — Surveillance / scanning
    'satan': 'Probe', 'ipsweep': 'Probe', 'portsweep': 'Probe', 'nmap': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L — Remote to Local
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L',
    'imap': 'R2L', 'ftp_write': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
    'spy': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'sendmail': 'R2L',
    'named': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
    # U2R — User to Root (privilege escalation)
    'buffer_overflow': 'U2R', 'rootkit': 'U2R', 'loadmodule': 'U2R',
    'perl': 'U2R', 'ps': 'U2R', 'xterm': 'U2R', 'sqlattack': 'U2R',
    # Normal
    'normal': 'Normal',
}

CLASS_MAPPING = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
CLASS_LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
