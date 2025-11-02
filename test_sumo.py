#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time

# Check SUMO_HOME
print("Checking SUMO_HOME environment variable...")
if 'SUMO_HOME' in os.environ:
    sumo_home = os.environ['SUMO_HOME']
    print(f"SUMO_HOME is set to: {sumo_home}")
    tools = os.path.join(sumo_home, 'tools')
    print(f"Tools path: {tools}")
    sys.path.append(tools)
else:
    print("ERROR: SUMO_HOME environment variable is not set!")
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Try to import traci
print("Trying to import traci...")
try:
    import traci
    from sumolib import checkBinary
    print("Successfully imported traci and sumolib")
except ImportError as e:
    print(f"ERROR: Failed to import traci or sumolib: {e}")
    sys.exit("Failed to import SUMO libraries")

# Check if SUMO binaries exist
print("Checking SUMO binaries...")
try:
    sumo_binary = checkBinary('sumo')
    print(f"SUMO binary found at: {sumo_binary}")
    
    sumo_gui_binary = checkBinary('sumo-gui')
    print(f"SUMO-GUI binary found at: {sumo_gui_binary}")
except Exception as e:
    print(f"ERROR: Failed to find SUMO binaries: {e}")
    sys.exit("Failed to find SUMO binaries")

# Check if configuration files exist
print("Checking configuration files...")
if os.path.exists("simulation.sumocfg"):
    print("simulation.sumocfg found")
else:
    print("ERROR: simulation.sumocfg not found!")

if os.path.exists("network.net.xml"):
    print("network.net.xml found")
else:
    print("ERROR: network.net.xml not found!")

if os.path.exists("routes.rou.xml"):
    print("routes.rou.xml found")
else:
    print("ERROR: routes.rou.xml not found!")

# Try to start a simple simulation
print("\nAttempting to start a simple SUMO simulation...")
try:
    # Start SUMO with a small timeout
    traci.start([sumo_binary, "-c", "simulation.sumocfg", "--no-step-log", "true"])
    print("SUMO started successfully")
    
    # Run for a few steps
    print("Running simulation for 10 steps...")
    for step in range(10):
        print(f"Step {step}")
        traci.simulationStep()
        time.sleep(0.1)  # Small delay to see output
    
    # Get vehicle IDs
    vehicle_ids = traci.vehicle.getIDList()
    print(f"Vehicles in simulation: {vehicle_ids}")
    
    # Check for ego vehicle
    if "ego" in vehicle_ids:
        print("Ego vehicle found in simulation")
        
        # Get ego vehicle data
        ego_speed = traci.vehicle.getSpeed("ego")
        ego_pos = traci.vehicle.getPosition("ego")
        print(f"Ego vehicle speed: {ego_speed} m/s")
        print(f"Ego vehicle position: {ego_pos}")
    else:
        print("WARNING: Ego vehicle not found in simulation!")
    
    # Close SUMO
    traci.close()
    print("SUMO simulation closed successfully")
    
except Exception as e:
    print(f"ERROR during simulation: {e}")
    sys.exit("Failed to run SUMO simulation")

print("\nTest completed successfully!") 