"""
Dynamic Train Dispatching Problem (DTSP) - Synthetic Data Generator

Generates realistic transit network data including:
- Network topology (stations, tracks, travel times)
- Origin-Destination (OD) demand matrices
- Operational parameters
- Time-varying demand patterns

Usage:
    python generate_dtsp_data.py --network_type circular --num_stations 20
"""

import numpy as np
import pandas as pd
import networkx as nx
import json
from datetime import datetime, timedelta
from pathlib import Path
import argparse


class DTSPDataGenerator:
    """Generate synthetic data for Dynamic Train Dispatching Problem"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
        
    def generate_network(self, network_type='circular', num_stations=15):
        """
        Generate transit network topology
        
        Args:
            network_type: 'circular' (loop line like Yamanote), 
                         'linear' (subway line), 
                         'branched' (line with branches)
            num_stations: Number of stations
        
        Returns:
            NetworkX DiGraph with station and track attributes
        """
        G = nx.DiGraph()
        
        # Generate stations
        stations = []
        for i in range(num_stations):
            station_id = f"S{i:02d}"
            
            # Station attributes
            station_data = {
                'id': station_id,
                'name': self._generate_station_name(i),
                'platform_capacity': np.random.randint(300, 800),  # passengers
                'is_transfer': np.random.random() < 0.2,  # 20% are transfer stations
                'zone': 'urban' if i < num_stations * 0.7 else 'suburban',
                'coordinates': self._generate_coordinates(i, num_stations, network_type),
            }
            
            # Higher demand at transfer stations
            if station_data['is_transfer']:
                station_data['platform_capacity'] *= 1.5
                station_data['base_demand_multiplier'] = 1.8
            else:
                station_data['base_demand_multiplier'] = 1.0
            
            stations.append(station_data)
            G.add_node(station_id, **station_data)
        
        # Generate edges (tracks between stations)
        if network_type == 'circular':
            edges = self._generate_circular_edges(num_stations)
        elif network_type == 'linear':
            edges = self._generate_linear_edges(num_stations)
        elif network_type == 'branched':
            edges = self._generate_branched_edges(num_stations)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        for origin_idx, dest_idx in edges:
            origin = f"S{origin_idx:02d}"
            dest = f"S{dest_idx:02d}"
            
            # Calculate distance (simplified)
            orig_coords = G.nodes[origin]['coordinates']
            dest_coords = G.nodes[dest]['coordinates']
            distance = np.sqrt(
                (orig_coords[0] - dest_coords[0])**2 + 
                (orig_coords[1] - dest_coords[1])**2
            )
            
            # Track attributes
            track_data = {
                'distance_km': round(distance, 2),
                'avg_travel_time': round(distance * 2.5 + np.random.uniform(0.5, 1.5), 2),  # minutes
                'capacity_trains_per_hour': np.random.randint(25, 35),
                'reliability': round(np.random.uniform(0.92, 0.98), 3),
                'delay_std': round(np.random.uniform(0.3, 0.8), 2),  # minutes
                'track_type': 'double' if np.random.random() > 0.1 else 'single',
            }
            
            G.add_edge(origin, dest, **track_data)
        
        return G, stations
    
    def _generate_station_name(self, idx):
        """Generate realistic-sounding station names"""
        prefixes = ['Shin', 'Nishi', 'Higashi', 'Minami', 'Kita', 'Chuo', 'Kami', 'Shimo']
        suffixes = ['bashi', 'cho', 'gawa', 'yama', 'oka', 'sawa', 'machi', 'dori']
        
        if idx % 5 == 0:  # Major stations
            names = ['Shibuya', 'Shinjuku', 'Tokyo', 'Ikebukuro', 'Ueno', 'Shinagawa']
            return names[idx // 5] if idx // 5 < len(names) else f"Station_{idx:02d}"
        else:
            return f"{np.random.choice(prefixes)}{np.random.choice(suffixes)}"
    
    def _generate_coordinates(self, idx, total, network_type):
        """Generate station coordinates"""
        if network_type == 'circular':
            angle = 2 * np.pi * idx / total
            radius = 10  # km
            x = radius * np.cos(angle) + np.random.normal(0, 0.2)
            y = radius * np.sin(angle) + np.random.normal(0, 0.2)
        elif network_type == 'linear':
            x = idx * 1.5 + np.random.normal(0, 0.2)
            y = np.random.normal(0, 0.3)
        else:  # branched
            if idx < total * 0.6:
                x = idx * 1.5
                y = 0
            else:
                branch_idx = idx - int(total * 0.6)
                x = total * 0.6 * 1.5 + branch_idx * 0.8
                y = branch_idx * 1.2
        
        return (round(x, 3), round(y, 3))
    
    def _generate_circular_edges(self, n):
        """Generate edges for circular line (e.g., Yamanote)"""
        edges = [(i, (i + 1) % n) for i in range(n)]  # Forward direction
        edges += [((i + 1) % n, i) for i in range(n)]  # Reverse direction
        return edges
    
    def _generate_linear_edges(self, n):
        """Generate edges for linear line"""
        edges = [(i, i + 1) for i in range(n - 1)]  # Forward
        edges += [(i + 1, i) for i in range(n - 1)]  # Reverse
        return edges
    
    def _generate_branched_edges(self, n):
        """Generate edges for branched line"""
        main_line = int(n * 0.6)
        edges = [(i, i + 1) for i in range(main_line - 1)]
        edges += [(i + 1, i) for i in range(main_line - 1)]
        
        # Add branch
        for i in range(main_line, n - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        
        # Connect main line to branch
        branch_point = main_line - 1
        edges.append((branch_point, main_line))
        edges.append((main_line, branch_point))
        
        return edges
    
    def generate_od_matrix(self, stations, simulation_date='2024-01-15'):
        """
        Generate Origin-Destination demand matrix
        
        Returns:
            DataFrame with columns: origin, destination, time_period, avg_demand, demand_std
        """
        od_data = []
        
        time_periods = {
            'early_morning': (5, 7),    # 5-7am
            'morning_peak': (7, 9),     # 7-9am
            'midday': (9, 17),          # 9am-5pm
            'evening_peak': (17, 20),   # 5-8pm
            'night': (20, 24)           # 8pm-12am
        }
        
        for origin in stations:
            for destination in stations:
                if origin['id'] == destination['id']:
                    continue
                
                for period_name, (start_hour, end_hour) in time_periods.items():
                    # Base demand (gravity model-ish)
                    distance = self._calculate_distance(
                        origin['coordinates'], 
                        destination['coordinates']
                    )
                    
                    # Demand decreases with distance
                    base_demand = 100 / (1 + distance / 5)
                    
                    # Apply multipliers
                    base_demand *= origin['base_demand_multiplier']
                    base_demand *= destination['base_demand_multiplier']
                    
                    # Time period adjustments
                    if period_name == 'morning_peak':
                        # Commute to city center (lower station IDs)
                        if int(destination['id'][1:]) < int(origin['id'][1:]):
                            base_demand *= 2.5
                        else:
                            base_demand *= 0.5
                    elif period_name == 'evening_peak':
                        # Commute from city center
                        if int(destination['id'][1:]) > int(origin['id'][1:]):
                            base_demand *= 2.5
                        else:
                            base_demand *= 0.5
                    elif period_name == 'midday':
                        base_demand *= 0.6
                    elif period_name == 'early_morning':
                        base_demand *= 0.3
                    elif period_name == 'night':
                        base_demand *= 0.4
                    
                    # Add noise
                    demand = max(0, base_demand + np.random.normal(0, base_demand * 0.2))
                    demand_std = demand * 0.3  # 30% standard deviation
                    
                    od_data.append({
                        'origin': origin['id'],
                        'destination': destination['id'],
                        'time_period': period_name,
                        'start_hour': start_hour,
                        'end_hour': end_hour,
                        'avg_demand': round(demand, 2),
                        'demand_std': round(demand_std, 2),
                        'date': simulation_date
                    })
        
        return pd.DataFrame(od_data)
    
    def _calculate_distance(self, coord1, coord2):
        """Euclidean distance between coordinates"""
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    def generate_hourly_demand_profile(self, od_matrix):
        """
        Generate detailed hourly demand patterns
        
        Returns:
            DataFrame with hourly breakdown
        """
        hourly_data = []
        
        for _, row in od_matrix.iterrows():
            period_hours = row['end_hour'] - row['start_hour']
            
            for hour in range(row['start_hour'], row['end_hour']):
                # Distribute demand across hours within period
                if row['time_period'] == 'morning_peak':
                    # Peak at 8am
                    if hour == 8:
                        hour_multiplier = 1.5
                    elif hour == 7:
                        hour_multiplier = 1.2
                    else:
                        hour_multiplier = 0.8
                elif row['time_period'] == 'evening_peak':
                    # Peak at 6pm
                    if hour == 18:
                        hour_multiplier = 1.5
                    elif hour in [17, 19]:
                        hour_multiplier = 1.2
                    else:
                        hour_multiplier = 0.8
                else:
                    hour_multiplier = 1.0 / period_hours
                
                hourly_demand = row['avg_demand'] * hour_multiplier
                
                hourly_data.append({
                    'origin': row['origin'],
                    'destination': row['destination'],
                    'hour': hour,
                    'time_period': row['time_period'],
                    'demand': round(hourly_demand, 2),
                    'demand_std': row['demand_std'] * hour_multiplier,
                    'date': row['date']
                })
        
        return pd.DataFrame(hourly_data)
    
    def generate_operational_parameters(self, num_stations):
        """
        Generate operational constraints and parameters
        """
        params = {
            'network': {
                'num_stations': num_stations,
                'min_headway_minutes': 1.5,  # Minimum time between trains (safety)
                'max_headway_minutes': 15.0,  # Maximum allowed gap
                'signal_system': 'modern',  # or 'legacy'
            },
            'trains': {
                'fleet_size': int(num_stations * 1.5),  # Total available trains
                'capacity_passengers': 1200,  # Per train
                'standing_capacity': 400,  # Additional standing room
                'comfortable_occupancy': 0.8,  # 80% capacity is comfortable
                'max_speed_kmh': 80,
                'acceleration_rate': 1.0,  # m/s^2
                'deceleration_rate': 1.2,  # m/s^2
            },
            'operations': {
                'dwell_time_base': 0.5,  # Base station stop time (minutes)
                'dwell_time_per_100_pax': 0.3,  # Additional time per 100 passengers
                'max_dwell_time': 2.0,  # Maximum station stop
                'turnaround_time': 5.0,  # Time to reverse at terminal
            },
            'costs': {
                'cost_per_train_km': 150,  # Yen
                'cost_per_train_hour': 5000,  # Yen
                'passenger_time_value': 30,  # Yen per passenger-minute
                'crowding_discomfort_cost': 50,  # Yen per overcrowded passenger
            },
            'service_levels': {
                'target_on_time_percentage': 0.90,
                'max_acceptable_wait_time': 15,  # minutes
                'target_occupancy_rate': 0.70,
                'max_safe_occupancy': 1.5,  # 150% capacity (dangerous)
            }
        }
        
        return params
    
    def save_dataset(self, output_dir='data/dtsp_synthetic'):
        """
        Generate and save complete DTSP dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating DTSP synthetic dataset...")
        print("=" * 60)
        
        # 1. Generate network
        print("\n1. Generating network topology...")
        network, stations = self.generate_network(
            network_type='circular',  # Can change to 'linear' or 'branched'
            num_stations=15
        )
        
        # Save network as GML (graph format)
        network_path = output_path / 'network.gml'
        nx.write_gml(network, network_path)
        print(f"   ✓ Saved network: {network_path}")
        
        # Save stations as CSV
        stations_df = pd.DataFrame(stations)
        stations_path = output_path / 'stations.csv'
        stations_df.to_csv(stations_path, index=False)
        print(f"   ✓ Saved stations: {stations_path}")
        
        # 2. Generate OD matrix
        print("\n2. Generating OD demand matrix...")
        od_matrix = self.generate_od_matrix(stations)
        od_path = output_path / 'od_matrix.csv'
        od_matrix.to_csv(od_path, index=False)
        print(f"   ✓ Saved OD matrix: {od_path}")
        print(f"   ✓ Total OD pairs: {len(od_matrix)}")
        
        # 3. Generate hourly demand
        print("\n3. Generating hourly demand profiles...")
        hourly_demand = self.generate_hourly_demand_profile(od_matrix)
        hourly_path = output_path / 'hourly_demand.csv'
        hourly_demand.to_csv(hourly_path, index=False)
        print(f"   ✓ Saved hourly demand: {hourly_path}")
        
        # 4. Generate operational parameters
        print("\n4. Generating operational parameters...")
        params = self.generate_operational_parameters(len(stations))
        params_path = output_path / 'operational_params.json'
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"   ✓ Saved parameters: {params_path}")
        
        # 5. Generate summary statistics
        print("\n5. Generating summary statistics...")
        summary = self._generate_summary(network, od_matrix, stations, params)
        summary_path = output_path / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✓ Saved summary: {summary_path}")
        
        # 6. Generate README
        print("\n6. Generating README...")
        readme_path = output_path / 'README.md'
        self._generate_readme(readme_path, summary)
        print(f"   ✓ Saved README: {readme_path}")
        
        print("\n" + "=" * 60)
        print(f"✓ Dataset generation complete!")
        print(f"✓ Output directory: {output_path.absolute()}")
        print("\nDataset includes:")
        print("  - network.gml (Network topology)")
        print("  - stations.csv (Station attributes)")
        print("  - od_matrix.csv (Origin-Destination demand)")
        print("  - hourly_demand.csv (Hourly demand breakdown)")
        print("  - operational_params.json (System parameters)")
        print("  - dataset_summary.json (Statistics)")
        print("  - README.md (Documentation)")
        
        return output_path
    
    def _generate_summary(self, network, od_matrix, stations, params):
        """Generate dataset summary statistics"""
        return {
            'generation_date': datetime.now().isoformat(),
            'seed': self.seed,
            'network': {
                'num_stations': len(stations),
                'num_tracks': network.number_of_edges(),
                'network_type': 'circular',
                'total_length_km': round(sum(
                    data['distance_km'] 
                    for _, _, data in network.edges(data=True)
                ), 2),
                'avg_inter_station_distance': round(np.mean([
                    data['distance_km'] 
                    for _, _, data in network.edges(data=True)
                ]), 2)
            },
            'demand': {
                'total_od_pairs': len(od_matrix),
                'avg_daily_demand_per_pair': round(od_matrix['avg_demand'].mean(), 2),
                'peak_demand_period': 'morning_peak',
                'total_daily_passengers': round(od_matrix['avg_demand'].sum(), 0),
            },
            'operations': params
        }
    
    def _generate_readme(self, path, summary):
        """Generate README documentation"""
        readme_content = f"""# DTSP Synthetic Dataset

## Overview
This dataset contains synthetic data for a Dynamic Train Dispatching Problem (DTSP).

**Generated:** {summary['generation_date']}  
**Seed:** {summary['seed']}

## Network Characteristics
- **Stations:** {summary['network']['num_stations']}
- **Track segments:** {summary['network']['num_tracks']}
- **Network type:** {summary['network']['network_type']}
- **Total length:** {summary['network']['total_length_km']} km
- **Average inter-station distance:** {summary['network']['avg_inter_station_distance']} km

## Demand Characteristics
- **OD pairs:** {summary['demand']['total_od_pairs']}
- **Estimated daily passengers:** {summary['demand']['total_daily_passengers']:,.0f}
- **Peak period:** {summary['demand']['peak_demand_period']}

## Files

### 1. `network.gml`
Network topology in GML format (Graph Modeling Language).
- Nodes: Stations with attributes (capacity, type, location)
- Edges: Track segments with travel times, capacity, reliability

### 2. `stations.csv`
Station-level attributes:
- `id`: Station identifier
- `name`: Station name
- `platform_capacity`: Maximum passengers on platform
- `is_transfer`: Boolean indicating transfer station
- `zone`: Urban or suburban
- `coordinates`: (x, y) location

### 3. `od_matrix.csv`
Origin-Destination demand matrix by time period:
- `origin`, `destination`: Station IDs
- `time_period`: Morning peak, evening peak, midday, etc.
- `avg_demand`: Average passengers per period
- `demand_std`: Standard deviation of demand

### 4. `hourly_demand.csv`
Hourly breakdown of OD demand:
- `origin`, `destination`: Station IDs
- `hour`: Hour of day (0-23)
- `demand`: Expected passengers for that hour
- `demand_std`: Demand uncertainty

### 5. `operational_params.json`
System-wide operational parameters:
- Network constraints (min/max headway)
- Train specifications (capacity, speed)
- Cost parameters
- Service level targets

## Usage

```python
import pandas as pd
import networkx as nx
import json

# Load network
network = nx.read_gml('network.gml')

# Load demand
od_matrix = pd.read_csv('od_matrix.csv')
hourly_demand = pd.read_csv('hourly_demand.csv')

# Load parameters
with open('operational_params.json') as f:
    params = json.load(f)

# Load stations
stations = pd.read_csv('stations.csv')
```

## Time Periods

- **Early Morning**: 5:00 AM - 7:00 AM
- **Morning Peak**: 7:00 AM - 9:00 AM
- **Midday**: 9:00 AM - 5:00 PM
- **Evening Peak**: 5:00 PM - 8:00 PM
- **Night**: 8:00 PM - 12:00 AM

## Notes

- Demand follows realistic patterns (commute flows, peak periods)
- Network includes transfer stations with higher capacity
- All travel times include realistic variance
- Safety constraints (minimum headway) are included
"""
        
        with open(path, 'w') as f:
            f.write(readme_content)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic DTSP dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/dtsp_synthetic',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--num-stations',
        type=int,
        default=15,
        help='Number of stations in network'
    )
    parser.add_argument(
        '--network-type',
        type=str,
        choices=['circular', 'linear', 'branched'],
        default='circular',
        help='Type of network topology'
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = DTSPDataGenerator(seed=args.seed)
    output_path = generator.save_dataset(output_dir=args.output_dir)
    
    print(f"\n{'='*60}")
    print("Next steps:")
    print("1. Review the generated data in:", output_path)
    print("2. Use this data to build your DTSP simulation")
    print("3. Implement fitness function based on operational_params.json")
    print("4. Create seed strategies for ShinkaEvolve")


if __name__ == '__main__':
    main()