
# Forensic Data Analytics for Anomaly Detection in Evolving Networks Using Wireshark

## Overview
Networks are constantly changing, and keeping them secure means knowing exactly what is moving through them. I built this project to do exactly that—analyze network traffic and spot anomalies using Wireshark. By capturing and examining packet-level data, this system acts as a digital magnifying glass, helping to identify suspicious activities, unusual traffic patterns, and potential security threats before they escalate.

## Objectives
I set out to achieve a few specific goals with this project:
- Capture and analyze network traffic effectively using Wireshark.
- Detect abnormal network behavior and active security incidents.
- Perform hands-on forensic analysis on captured packets.
- Pull actionable insights from raw network traffic data.
- Improve overall network monitoring and threat detection capabilities.

## Technologies Used
- **Wireshark:** For network packet capture and deep-dive analysis.
- **Python:** The backbone for data processing and analytics.
- **Pandas:** To handle and manipulate the traffic data.
- **NumPy:** For the heavy lifting with numerical computations.
- **Matplotlib/Seaborn:** To visualize the data and make the findings easy to understand.

## Features
- Both real-time and offline packet analysis.
- Detection of unusual network traffic patterns that deviate from the norm.
- Protocol-based traffic inspection to see exactly what's talking to what.
- Clear visualizations of network statistics.
- Forensic investigation support to help trace back cybersecurity incidents.

## Project Workflow
Here is the step-by-step process I followed to make this work:
1. Capture network packets using Wireshark.
2. Export the packet data for analysis.
3. Clean and preprocess the raw network traffic data.
4. Apply anomaly detection techniques to find the outliers.
5. Visualize the findings and generate readable reports.
6. Identify potential threats and flag suspicious activities.

## Dataset
The dataset consists of network packet captures (PCAP files) collected using Wireshark. These files contain the raw details of the network, including:
- Source and destination IP addresses
- Protocol types
- Packet sizes
- Timestamps
- Network flow details

## Results
The project successfully identifies anomalous network behavior by analyzing traffic patterns and highlighting deviations from normal activity. It serves as an early warning system for potential cyber threats and provides solid support for digital forensic investigations.

## Future Enhancements
There is always room to make a security tool smarter. Here is what I am planning to add next:
- Machine learning-based anomaly detection to catch things simple rules might miss.
- Real-time alert generation so administrators know the moment a threat is detected.
- Integration with existing Intrusion Detection Systems (IDS).
- Advanced threat intelligence support to cross-reference flagged IPs.

## Installation
If you want to run this locally, here are the commands to get started:

```bash
git clone https://github.com/your-username/Forensic-Data-Analytics-for-Anomaly-Detection-in-Evolving-Networks-Using-Wireshark.git
cd Forensic-Data-Analytics-for-Anomaly-Detection-in-Evolving-Networks-Using-Wireshark
```
