import streamlit as st
import pandas as pd
import folium
import math
from io import StringIO
import streamlit.components.v1 as components
# from streamlit_folium import st_folium  
import geopandas
from pyproj import Transformer

fouteen_ad_geojson = geopandas.read_file(r"Dev Data Files\roman_empire_ad_14_provinces.geojson")

sixtynine_ad_geojson = geopandas.read_file(r"Dev Data Files\roman_empire_ad_69_provinces.geojson")

klokan_provinces_geojson = geopandas.read_file(r"Dev Data Files\klokan_provinces.geojson")

routes_roads_only = geopandas.read_file(r"Dev Data Files\routes_roads_only.geojson")

data = pd.read_csv(r"Test Inputs\Corrected data files\combined_data.csv", dtype={'id': int, 'hoard': int})

def convert_to_radii(raw_number, scale_factor=1):
    """
    Convert a raw number to a radius that respects the area principle.
    
    Parameters:
      raw_number (float): A raw number to convert.
      scale_factor (float): A scale factor to modulate the radius.
    
    Returns:
      float: A radius corresponding to the raw number.
    """
    area = raw_number
    radius = math.sqrt(area / math.pi) * scale_factor
    return radius



def extract_data_from_csv(file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file)
    return data

def transform_coordinates(x, y, input_crs, output_crs='EPSG:4326'):
    """Transforms coordinates from one CRS to another."""
    transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
    return transformer.transform(x, y)

# Example: Transforming from a hypothetical CRS to WGS84
# x, y = some_x_coordinate, some_y_coordinate
# transformed_x, transformed_y = transform_coordinates(x, y, 'EPSG:XXXX') #replace EPSG:XXXX with your data's CRS.

def aggregate_same_location_circles(data):
    """
    Aggregate circles at the exact same location.
    
    Parameters:
      data (DataFrame): DataFrame containing coin data with latitude, longitude, and quantity.
    
    Returns:
      DataFrame: DataFrame where circles at the same location are combined.
    """
    # Create a copy of the data to avoid modifying the original
    aggregated_data = data.copy()
    
    # Group by location (latitude, longitude) and denomination
    location_grouped = aggregated_data.groupby(['latitude', 'longitude', 'denomination'])
    
    # Create a new DataFrame to store the location-aggregated results
    location_result = []
    
    # Process each location group
    for (lat, lon, denom), group in location_grouped:
        # Sum the quantities for the same location and denomination
        total_quantity = group['quantity'].sum()
        
        # Take the first row as a template for the aggregated row
        template_row = group.iloc[0].copy()
        
        # Update the quantity with the total
        template_row['quantity'] = total_quantity
        
        # Create a combined hoard name if there are multiple different hoards
        if group['hoardName'].nunique() > 1:
            hoard_names = group['hoardName'].unique()
            template_row['hoardName'] = f"Aggregated: {', '.join(hoard_names)}"
            
            # Create a detailed tooltip with individual quantities
            tooltip_parts = []
            for _, hoard_row in group.iterrows():
                tooltip_parts.append(f"{hoard_row['hoardName']}: {int(hoard_row['quantity'])}")
            template_row['tooltip'] = f"Total: {int(total_quantity)}\n" + "\n".join(tooltip_parts)
        else:
            template_row['tooltip'] = f"{template_row['hoardName']}\n{int(total_quantity)}"
        
        location_result.append(template_row)
    
    # Convert the location result list to a DataFrame
    return pd.DataFrame(location_result)

def aggregate_by_distance_circles(data, distance_threshold=10000, single_size=10000, relative_scale=1.0):
    """
    Aggregate circles that are within a certain distance of each other.
    
    Parameters:
      data (DataFrame): DataFrame containing coin data with latitude, longitude, and quantity.
      distance_threshold (float): Maximum distance (meters) between circles to be aggregated.
      single_size (float): Base radius size for circles.
      relative_scale (float): Scale factor for circle sizes.
    
    Returns:
      DataFrame: DataFrame where nearby circles are combined.
    """
    # Create a copy of the data to avoid modifying the original
    aggregated_data = data.copy()
    
    # Define haversine distance function
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points in meters."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r
    
    # Create a list to track which rows have been aggregated
    aggregated_indices = set()
    final_result = []
    
    # Sort by quantity in descending order to prioritize larger circles
    sorted_data = aggregated_data.sort_values('quantity', ascending=False).reset_index(drop=True)
    
    # For each circle, check if it meets the distance-based aggregation criteria
    for i, row1 in sorted_data.iterrows():
        if i in aggregated_indices:
            continue
        
        # Find all circles that meet the distance-based aggregation criteria
        circles_to_aggregate = []
        aggregated_quantity = 0
        aggregated_details = []
        
        for j, row2 in sorted_data.iterrows():
            if i == j or j in aggregated_indices:
                continue
                
            # Only consider circles of the same denomination
            if row1['denomination'] != row2['denomination']:
                continue
                
            # Calculate distance between circle centers
            try:
                distance = haversine_distance(
                    float(row1['latitude']), float(row1['longitude']),
                    float(row2['latitude']), float(row2['longitude'])
                )
                
                # Check if the circle meets the distance-based aggregation criteria
                if distance <= distance_threshold:
                    circles_to_aggregate.append(j)
                    aggregated_quantity += float(row2['quantity'])
                    aggregated_details.append((row2['hoardName'], float(row2['quantity'])))
            except (ValueError, TypeError):
                # Skip if conversion to float fails
                continue
        
        # If we found circles to aggregate, combine them
        if circles_to_aggregate:
            # Create a new aggregated row
            new_row = row1.copy()
            quantity1 = float(row1['quantity'])
            total_quantity = quantity1 + aggregated_quantity
            new_row['quantity'] = total_quantity
            
            # Update the hoard name to indicate aggregation
            hoard_names = [row1['hoardName']]
            for idx in circles_to_aggregate:
                hoard_names.append(sorted_data.iloc[idx]['hoardName'])
                aggregated_indices.add(idx)
            
            new_row['hoardName'] = f"Aggregated: {', '.join(set(hoard_names))}"
            
            # Create a detailed tooltip with individual quantities
            tooltip_parts = [f"{row1['hoardName']}: {int(quantity1)}"]
            for hoard_name, quantity in aggregated_details:
                tooltip_parts.append(f"{hoard_name}: {int(quantity)}")
            new_row['tooltip'] = f"Total: {int(total_quantity)}\n" + "\n".join(tooltip_parts)
            
            final_result.append(new_row)
            aggregated_indices.add(i)
        else:
            # If this circle doesn't meet any aggregation criteria, keep it as is
            if 'tooltip' not in row1:
                row1['tooltip'] = f"{row1['hoardName']}\n{int(float(row1['quantity']))}"
            final_result.append(row1)
    
    # Return the final aggregated DataFrame
    return pd.DataFrame(final_result)

def aggregate_enveloped_circles(data, single_size=10000, relative_scale=1.0):
    """
    Aggregate circles where one completely envelops another.
    
    Parameters:
      data (DataFrame): DataFrame containing coin data with latitude, longitude, and quantity.
      single_size (float): Base radius size for circles.
      relative_scale (float): Scale factor for circle sizes.
    
    Returns:
      DataFrame: DataFrame where enveloped circles are combined with their super circles.
    """
    # Create a copy of the data to avoid modifying the original
    aggregated_data = data.copy()
    
    # Define haversine distance function
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points in meters."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r
    
    # Create a list to track which rows have been aggregated
    aggregated_indices = set()
    final_result = []
    
    # Sort by quantity in descending order to prioritize larger circles
    sorted_data = aggregated_data.sort_values('quantity', ascending=False).reset_index(drop=True)
    
    # For each circle, check if it envelops any others
    for i, row1 in sorted_data.iterrows():
        if i in aggregated_indices:
            continue
        
        try:
            # Calculate radius of this circle
            quantity1 = float(row1['quantity'])
            radius1 = math.sqrt(quantity1) * single_size * relative_scale
            
            # Find all circles that this one might envelop
            enveloped_circles = []
            enveloped_quantity = 0
            enveloped_details = []
            
            for j, row2 in sorted_data.iterrows():
                if i == j or j in aggregated_indices:
                    continue
                    
                # Only consider circles of the same denomination
                if row1['denomination'] != row2['denomination']:
                    continue
                
                try:
                    # Calculate radius of the other circle
                    quantity2 = float(row2['quantity'])
                    radius2 = math.sqrt(quantity2) * single_size * relative_scale
                    
                    # Calculate distance between circle centers
                    distance = haversine_distance(
                        float(row1['latitude']), float(row1['longitude']),
                        float(row2['latitude']), float(row2['longitude'])
                    )
                    
                    # If the distance is less than the difference of the radii, the smaller circle is completely inside the larger one
                    if distance + radius2 <= radius1:
                        enveloped_circles.append(j)
                        enveloped_quantity += quantity2
                        enveloped_details.append((row2['hoardName'], quantity2))
                except (ValueError, TypeError):
                    # Skip if conversion to float fails
                    continue
            
            # If we found enveloped circles, aggregate them
            if enveloped_circles:
                # Create a new aggregated row
                new_row = row1.copy()
                total_quantity = quantity1 + enveloped_quantity
                new_row['quantity'] = total_quantity
                
                # Update the hoard name to indicate aggregation
                hoard_names = [row1['hoardName']]
                for idx in enveloped_circles:
                    hoard_names.append(sorted_data.iloc[idx]['hoardName'])
                    aggregated_indices.add(idx)
                
                new_row['hoardName'] = f"Aggregated: {', '.join(set(hoard_names))}"
                
                # Create a detailed tooltip with individual quantities
                tooltip_parts = [f"{row1['hoardName']}: {int(quantity1)}"]
                for hoard_name, quantity in enveloped_details:
                    tooltip_parts.append(f"{hoard_name}: {int(quantity)}")
                new_row['tooltip'] = f"Total: {int(total_quantity)}\n" + "\n".join(tooltip_parts)
                
                final_result.append(new_row)
                aggregated_indices.add(i)
            else:
                # If this circle doesn't envelop any others, keep it as is
                if 'tooltip' not in row1:
                    row1['tooltip'] = f"{row1['hoardName']}\n{int(quantity1)}"
                final_result.append(row1)
        except (ValueError, TypeError):
            # If conversion to float fails, keep the original row
            if 'tooltip' not in row1:
                try:
                    row1['tooltip'] = f"{row1['hoardName']}\n{int(float(row1['quantity']))}"
                except (ValueError, TypeError):
                    row1['tooltip'] = f"{row1['hoardName']}\nQuantity error"
            final_result.append(row1)
    
    # Return the final aggregated DataFrame
    return pd.DataFrame(final_result)

def aggregate_overlapping_circles(data, single_size=10000, relative_scale=1.0, aggregate_by_distance=False, distance_threshold=10000, aggregate_enveloped=False):
    """
    Master function to aggregate circles based on user-selected criteria.
    
    Parameters:
      data (DataFrame): DataFrame containing coin data with latitude, longitude, and quantity.
      single_size (float): Base radius size for circles, same as used in display.
      relative_scale (float): Scale factor for circle sizes, same as used in display.
      aggregate_by_distance (bool): Whether to aggregate circles within a certain distance of each other.
      distance_threshold (float): Maximum distance (in meters) between circle centers for aggregation.
      aggregate_enveloped (bool): Whether to aggregate circles that are completely enveloped by larger ones.
    
    Returns:
      DataFrame: Aggregated DataFrame where circles are combined according to the selected criteria.
    """
    # Always aggregate circles at the exact same location
    result = aggregate_same_location_circles(data)
    
    # Apply distance-based aggregation if requested
    if aggregate_by_distance:
        result = aggregate_by_distance_circles(result, distance_threshold, single_size, relative_scale)
    
    # Apply envelopment-based aggregation if requested
    if aggregate_enveloped:
        result = aggregate_enveloped_circles(result, single_size, relative_scale)
    
    return result

def create_map(data, scale_factor=3, zoom_level=5, fill_color="#708090", stroke_color="#708090", 
               opacity=0.7, weight=0.5, base_tile="OpenStreetMap", initial_center=None, initial_zoom=None,
               dare_fill_color="#f2f2f2", dare_stroke_color="#f2f2f2", dare_weight=1, dare_fill_opacity=0.3,
               route_color="#ffa500", route_weight=2.0, cutoff=0, icon_size=30, single_size=10000, 
               aureus_stroke_color="#FFD700", aureus_fill_color="#FFD700", aureus_stroke_weight=1.0, aureus_stroke_opacity=1.0, aureus_fill_opacity=0.7,
               denarius_stroke_color="#C0C0C0", denarius_fill_color="#C0C0C0", denarius_stroke_weight=1.0, denarius_stroke_opacity=1.0, denarius_fill_opacity=0.7,
               relative_scale=1.0, aggregate_by_distance=False, distance_threshold=10000, aggregate_enveloped=False):
    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    # If an initial_center is provided, use it; otherwise, compute the center from the data.
    if initial_center:
        location = initial_center   
    else:
        location = [center_lat, center_lon]
    # Similarly override zoom if provided.
    if initial_zoom is not None:
        zoom = initial_zoom
    else:
        zoom = zoom_level

    # Create a copy of the data to avoid modifying the original
    data_to_plot = data.copy()
    
    # Apply circle aggregation if requested
    if aggregate_by_distance or aggregate_enveloped:
        try:
            data_to_plot = aggregate_overlapping_circles(
                data_to_plot, 
                single_size=single_size, 
                relative_scale=relative_scale,
                aggregate_by_distance=aggregate_by_distance,
                distance_threshold=distance_threshold,
                aggregate_enveloped=aggregate_enveloped
            )
        except Exception as e:
            print(f"Aggregation error: {e}")
            # If aggregation fails, continue with the original data
            pass

    m = folium.Map(location=location, zoom_start=zoom, tiles=None)

    # Add tile layers with an option to switch between them
    if base_tile == "OpenStreetMap":
        osm_show = True
        imperium_show = False
        orbis_show = False
        cawm_show = False
    elif base_tile == "Imperium":
        osm_show = False
        imperium_show = True
        orbis_show = False
        cawm_show = False
    elif base_tile == "Orbis":
        osm_show = False
        imperium_show = False
        orbis_show = True
        cawm_show = False
    elif base_tile == "CAWM":
        osm_show = False
        imperium_show = False
        orbis_show = False
        cawm_show = True


    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=osm_show).add_to(m)
    
    folium.TileLayer(
        tiles="//dh.gu.se/tiles/imperium/{z}/{x}/{y}.png",
        name="Imperium",
        attr="Tiles: github.com/klokantech/roman-empire/",
        max_zoom=15,
        show=imperium_show
    ).add_to(m)

    folium.TileLayer(
        tiles="//d3msn78fivoryj.cloudfront.net/orbis_tiles/{z}/{x}/{y}.png",
        name="Orbis",
        attr="Tiles: Stanford ORBIS",
        max_zoom=15,
        show=orbis_show
    ).add_to(m)

    folium.TileLayer(
        tiles="//cawm.lib.uiowa.edu/tiles/{z}/{x}/{y}.png",
        name="CAWM",
        attr="Tiles: CAWM",
        max_zoom=15,
        show=cawm_show
    ).add_to(m)

    # Overlay control

    global fouteen_bc_geojson
    
    global sixtynine_ad_geojson

    global klokan_provinces_geojson

    fourteen_ad_group = folium.FeatureGroup(name="14AD Provinces", overlay=True, control=True, show=False)

    sixtynine_ad_group = folium.FeatureGroup(name="69AD Provinces", overlay=True, control=True, show=False)

    dare_group = folium.FeatureGroup(name="DARE Provinces", overlay=True, control=True, show=True)

    routes_group = folium.FeatureGroup(name="Routes", overlay=True, control=True, show=True)

    coins = folium.FeatureGroup(name="Coins", overlay=True, control=True, show=True)

    # Feature group styling

    def style_fourteen_ad(feature):
        return {
            'fillColor': '#ff0000',  # Red color for 14AD Provinces
            'color': '#ff0000',      # Red border
            'weight': 1,
            'fillOpacity': 0.5
        }

    def style_sixtynine_ad(feature):
        return {
            'fillColor': '#00ff00',  # Green color for 69AD Provinces
            'color': '#00ff00',      # Green border
            'weight': 1,
            'fillOpacity': 0.5
        }

    def style_dare(feature):
        return {
            'fillColor': dare_fill_color,      # user-selected fill color for DARE Provinces
            'color': dare_stroke_color,          # user-selected border color for DARE Provinces
            'weight': dare_weight,
            'fillOpacity': dare_fill_opacity
        }

    def style_routes(feature):
        return {
            'color': route_color,      # Use selected route color
            'weight': route_weight     # Use selected route weight
        }

    # Add GeoJSON data to the feature groups
    folium.GeoJson(fouteen_ad_geojson, style_function=style_fourteen_ad).add_to(fourteen_ad_group)
    folium.GeoJson(sixtynine_ad_geojson, style_function=style_sixtynine_ad).add_to(sixtynine_ad_group)
    folium.GeoJson(klokan_provinces_geojson, style_function=style_dare).add_to(dare_group)
    folium.GeoJson(routes_roads_only, style_function=style_routes).add_to(routes_group)

    # List to record rows that produce an error
    error_rows = []

    # New helper function to add points
    def add_point_to_map(row, feature_group, cutoff, scale_factor, fill_color, stroke_color, opacity, weight, single_size, 
                         aureus_stroke_color, aureus_fill_color, aureus_stroke_weight, aureus_stroke_opacity, aureus_fill_opacity,
                         denarius_stroke_color, denarius_fill_color, denarius_stroke_weight, denarius_stroke_opacity, denarius_fill_opacity,
                         relative_scale):
        """Add a point to the map with conditional marker type based on cutoff"""
        try:
            # Convert values - may throw ValueError
            quantity = float(row['quantity'])
            quantity_display = int(quantity)

            # lon, lat = transform_coordinates(float(row['longitude']), float(row['latitude']), input_crs='EPSG:4326', output_crs='EPSG:3857')

            lat = float(row['latitude'])
            lon = float(row['longitude'])
            hoard_name = row['hoardName']
            
            # Use custom tooltip if available, otherwise create a default one
            if 'tooltip' in row:
                tooltip_text = row['tooltip']
            else:
                tooltip_text = f"{hoard_name}\n{quantity_display}"
            
            if quantity < cutoff and row['denomination'] == 'Aureus':
                # Create circle marker for Aureus
                folium.Circle(
                    location=[lat, lon],
                    popup=tooltip_text,
                    tooltip=tooltip_text,
                    color=aureus_stroke_color,
                    radius=single_size,
                    fill=True,
                    stroke=True,
                    fill_color=aureus_fill_color,
                    weight=aureus_stroke_weight,
                    fillOpacity=aureus_fill_opacity,
                    opacity=aureus_stroke_opacity
                ).add_to(feature_group)
            elif quantity < cutoff and row['denomination'] == 'Denarius':
                # Create circle marker for Denarius
                folium.Circle(
                    location=[lat, lon],
                    popup=tooltip_text,
                    tooltip=tooltip_text,
                    color=denarius_stroke_color,
                    radius=single_size,
                    fill=True,
                    stroke=True,
                    fill_color=denarius_fill_color,
                    weight=denarius_stroke_weight,
                    fillOpacity=denarius_fill_opacity,
                    opacity=denarius_stroke_opacity
                ).add_to(feature_group)
            else:
                # Create circle marker for values above cutoff
                scaled_radius = math.sqrt(quantity) * single_size * relative_scale
                
                if row['denomination'] == 'Aureus':
                    folium.Circle(
                        location=[lat, lon],
                        popup=tooltip_text,
                        tooltip=tooltip_text,
                        radius=scaled_radius,
                        fill_opacity=aureus_fill_opacity,
                        fill=True,
                        weight=aureus_stroke_weight,
                        color=aureus_stroke_color,
                        fillColor=aureus_fill_color,
                        opacity=denarius_stroke_opacity
                    ).add_to(feature_group)
                elif row['denomination'] == 'Denarius':
                    folium.Circle(
                        location=[lat, lon],
                        popup=tooltip_text,
                        tooltip=tooltip_text,
                        radius=scaled_radius,
                        fill_opacity=denarius_fill_opacity,
                        fill=True,
                        weight=denarius_stroke_weight,
                        color=denarius_stroke_color,
                        fillColor=denarius_fill_color,
                        opacity=denarius_stroke_opacity
                    ).add_to(feature_group)
                else:
                    # Default case for other denominations
                    folium.Circle(
                        location=[lat, lon],
                        popup=tooltip_text,
                        tooltip=tooltip_text,
                        radius=scaled_radius,
                        fill_opacity=opacity,
                        fill=True,
                        weight=weight,
                        color=stroke_color,
                        fillColor=fill_color
                    ).add_to(feature_group)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid data in row: {e}")

    # Modified marker creation loop
    for _, row in data_to_plot.iterrows():
        try:
            add_point_to_map(
                row=row,
                feature_group=coins,
                cutoff=cutoff,
                scale_factor=scale_factor,
                fill_color=fill_color,
                stroke_color=stroke_color,
                opacity=opacity,
                weight=weight,
                single_size=single_size,
                aureus_stroke_color=aureus_stroke_color,
                aureus_fill_color=aureus_fill_color,
                aureus_stroke_weight=aureus_stroke_weight,
                aureus_stroke_opacity=aureus_stroke_opacity,
                aureus_fill_opacity=aureus_fill_opacity,
                denarius_stroke_color=denarius_stroke_color,
                denarius_fill_color=denarius_fill_color,
                denarius_stroke_weight=denarius_stroke_weight,
                denarius_stroke_opacity=denarius_stroke_opacity,
                denarius_fill_opacity=denarius_fill_opacity,
                relative_scale=relative_scale
            )
        except ValueError:
            error_rows.append(row.to_dict())
            continue

    # Now add the feature groups to the map
    fourteen_ad_group.add_to(m)
    sixtynine_ad_group.add_to(m)
    dare_group.add_to(m)
    routes_group.add_to(m)

    # Add coins group and explicitly keep it in front
    coins.add_to(m)
    m.keep_in_front(coins)

    # Add a layer control to allow switching between tile layers
    folium.LayerControl(overlay=True, collapsed=False).add_to(m)
    
    # Return both the map and the error_rows list
    return m, error_rows

def save_map_with_print_option(m, filename='map.html', grayscale=False):
    
    root = m.get_root()

    root.render()

    # folium_head, folium_body, and folium_script are rendered from the map's root
    folium_head = root.header.render()
    folium_body = root.html.render()
    folium_script = root.script.render()

    map_id = m.get_name()

    easy_print_js = f"""
    <script src="https://cdn.jsdelivr.net/npm/leaflet-easyprint@2.1.9/dist/bundle.min.js"></script>
    <script>
        // Assume that the map is available via a global reference (adjust as needed)
        var map = {map_id};
        var printer = L.easyPrint({{
            title: 'Print map',
            position: 'topleft',
            exportOnly: true,
            sizeModes: ['Current', 'A4Portrait', 'A4Landscape']
        }}).addTo(map);
    </script>
    """

    # Add grayscale filter if enabled
    if grayscale:
        grayscale_css = """
        <style>
            .leaflet-tile-container {
                filter: grayscale(100%);
            }
        </style>
        """
    else:
        grayscale_css = ""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    {folium_head}
    {grayscale_css}
    </head>
    <body>
    {folium_body}
    </body>
    <script>
    {folium_script}
    </script>
    {easy_print_js}
    </html>
    """

    return html

    # with open(filename, 'w', encoding='utf-8') as fh:
    #     fh.write(html)

def get_map_html(m):
    return m._repr_html_()

st.set_page_config(layout="wide")
st.title("WYSIWYG CHRE Map Editor")
st.markdown("""
Upload hoard data from CHRE to get started.
""")

# Add a description of the aggregation feature
with st.expander("About Circle Aggregation"):
    st.markdown("""
    ### Circle Aggregation Features
    
    This tool offers three ways to aggregate circles:
    
    1. **Same Location Aggregation** (Always On): Combines circles that are at the exact same coordinates.
       - This is automatically applied to clean up the map.
       - Useful for combining multiple coins from the same hoard at the same location.
    
    2. **Aggregate by Distance**: Combines circles that are within a specified distance of each other. 
       - You can set the maximum distance between circle centers (in meters) for aggregation.
       - This is useful for combining nearby hoards that might represent the same historical event or location.
    
    3. **Aggregate Enveloped Circles**: Combines smaller circles that are completely contained within larger ones.
       - A smaller circle is considered "enveloped" when it's completely inside a larger circle.
       - This is useful for simplifying the map by combining smaller finds with larger ones in the same area.
    
    All options will only combine circles of the same denomination (Aureus with Aureus, Denarius with Denarius).
    
    The aggregated circles will display a tooltip showing the total quantity and a breakdown of the individual hoards that were combined.
    
    You can enable either distance aggregation, enveloped aggregation, or both together.
    """)

# Sidebar options for interactive map customization
st.sidebar.header("Map Options")
scale_factor = st.sidebar.slider("Scale Factor", min_value=0.1, max_value=5.0, value=4.0, step=0.1)
zoom_level = st.sidebar.slider("Zoom Level", min_value=1, max_value=10, value=5, step=1)
fill_color = st.sidebar.color_picker("Fill Color", value="#C5C5C5")
stroke_color = st.sidebar.color_picker("Stroke Color", value="#7F7F7F")
opacity = st.sidebar.slider("Opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
weight = st.sidebar.slider("Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
easyprint_enabled = st.sidebar.checkbox("Enable EasyPrint", value=True)

# New selectbox for choosing tile layer
tile_layer_option = st.sidebar.selectbox("Select Tile Layer", ["CAWM", "OpenStreetMap", "Imperium", "Orbis"])
hide_zero = st.sidebar.checkbox("Hide rows with zero quantity", value=True)

# New sidebar header and inputs for DARE Provinces styling
st.sidebar.header("DARE Provinces Styling")
dare_fill_color = st.sidebar.color_picker("DARE Fill Color", value="#D8D6D6")
dare_stroke_color = st.sidebar.color_picker("DARE Border Color", value="#626262")
dare_weight = st.sidebar.slider("DARE Weight", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
dare_fill_opacity = st.sidebar.slider("DARE Fill Opacity", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

# Greyscale
grayscale = st.sidebar.checkbox("Greyscale", value=True)

# Add to sidebar controls (after DARE Provinces Styling section)
st.sidebar.header("Routes Styling")
route_color = st.sidebar.color_picker("Route Color", value="#080808")  # Orange default
route_weight = st.sidebar.slider("Route Weight", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# Add circle marker controls
st.sidebar.header("Circle Marker Styling")
aureus_color = st.sidebar.color_picker("Aureus Circle Color", value="#FFD700")  # Gold color
denarius_color = st.sidebar.color_picker("Denarius Circle Color", value="#C0C0C0")  # Silver color
single_size = st.sidebar.slider("Circle Radius (meters)", min_value=1000, max_value=50000, value=10000, step=1000)
aureus_stroke_color = st.sidebar.color_picker("Aureus Circle Stroke Color", value="#080808")  # Gold color
denarius_stroke_color = st.sidebar.color_picker("Denarius Circle Stroke Color", value="#080808")  # Silver color

# Add new controls for aureus and denarius styling
st.sidebar.subheader("Aureus Styling")
aureus_stroke_weight = st.sidebar.slider("Aureus Stroke Weight", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
aureus_stroke_opacity = st.sidebar.slider("Aureus Stroke Opacity", min_value=0.1, max_value=1.0, value=0.80, step=0.1)
aureus_fill_opacity = st.sidebar.slider("Aureus Fill Opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

st.sidebar.subheader("Denarius Styling")
denarius_stroke_weight = st.sidebar.slider("Denarius Stroke Weight", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
denarius_stroke_opacity = st.sidebar.slider("Denarius Stroke Opacity", min_value=0.1, max_value=1.0, value=0.80, step=0.1)
denarius_fill_opacity = st.sidebar.slider("Denarius Fill Opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

# Add relative scaling control
st.sidebar.header("Size Scaling")
relative_scale = st.sidebar.slider("Relative Scale Factor", min_value=0.1, max_value=10.0, value=1.0, step=0.1, 
                                  help="Controls the relationship between point value and size. Lower values make size differences more dramatic.")

# Add circle aggregation options
st.sidebar.header("Circle Aggregation")
aggregate_by_distance = st.sidebar.checkbox("Aggregate by Distance", value=False, 
                                          help="Combine circles within a specified distance of each other")
if aggregate_by_distance:
    distance_threshold = st.sidebar.slider("Distance Threshold (meters)", min_value=1000, max_value=100000, value=10000, step=1000,
                                         help="Maximum distance between circle centers for aggregation")
else:
    distance_threshold = 10000  # Default value when not used

aggregate_enveloped = st.sidebar.checkbox("Aggregate Enveloped Circles", value=False,
                                        help="Combine smaller circles that are completely enveloped by larger ones")

# Add cutoff control to sidebar
cutoff = st.sidebar.number_input("Cutoff Value", min_value=0, value=2)

# icon_size = st.sidebar.slider("Icon Weight", min_value=1, max_value=60, value=6, step=1)

# File uploader for CSV input
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file using our helper function
        df = extract_data_from_csv(uploaded_file)
        st.subheader("CSV Preview")
        st.write(df.head())
        
        # Prepare a dataframe for map plotting based on the "hide_zero" toggle
        df_map = df.copy()
        if hide_zero:
            df_map = df_map[df_map['quantity'] != 0]
        
        # Ensure that the required columns exist in the CSV
        required_columns = ["latitude", "longitude", "hoardName", "quantity"]   
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must include the following columns: {required_columns}")
        else:
            # Determine initial center and zoom from session state if available,
            # otherwise use the mean of the latitudes/longitudes from df_map and the slider value.
            if "last_center" in st.session_state:
                init_center = st.session_state.last_center
            else:
                init_center = [df_map['latitude'].mean(), df_map['longitude'].mean()]

            if "last_zoom" in st.session_state:
                init_zoom = st.session_state.last_zoom
            else:
                init_zoom = zoom_level

            # Create the map using the helper function, passing in the initial view
            m, error_rows = create_map(
                df_map,
                scale_factor,
                zoom_level,
                fill_color,
                stroke_color,
                opacity,
                weight,
                base_tile=tile_layer_option,
                initial_center=init_center,
                initial_zoom=init_zoom,
                dare_fill_color=dare_fill_color,
                dare_stroke_color=dare_stroke_color,
                dare_weight=dare_weight,
                dare_fill_opacity=dare_fill_opacity,
                route_color=route_color,
                route_weight=route_weight,
                cutoff=cutoff,
                # icon_size=icon_size,
                single_size=single_size,
                aureus_stroke_color=aureus_stroke_color,
                aureus_fill_color=aureus_color,
                aureus_stroke_weight=aureus_stroke_weight,
                aureus_stroke_opacity=aureus_stroke_opacity,
                aureus_fill_opacity=aureus_fill_opacity,
                denarius_stroke_color=denarius_stroke_color,
                denarius_fill_color=denarius_color,
                denarius_stroke_weight=denarius_stroke_weight,
                denarius_stroke_opacity=denarius_stroke_opacity,
                denarius_fill_opacity=denarius_fill_opacity,
                relative_scale=relative_scale,
                aggregate_by_distance=aggregate_by_distance,
                distance_threshold=distance_threshold,
                aggregate_enveloped=aggregate_enveloped
            )

            # If EasyPrint is enabled, add the necessary javascript and plugin
            if easyprint_enabled:
                map_html = save_map_with_print_option(m, grayscale=grayscale)
            else:
                map_html = m.get_root().render()

            st.subheader("Generated Map")
            st.components.v1.html(map_html, height=1000)

            # Provide a download button for the generated map.html using map_html content
            st.download_button("Download Map HTML", data=map_html, file_name="map.html", mime="text/html")
            
            # Display error rows in the specified format
            if error_rows:
                error_details = []
                for er in error_rows:
                    id_val = er.get('id', 'N/A')
                    coin_count_val = er.get('coinCount', 'N/A')
                    error_details.append(f"\n {id_val}, {coin_count_val}")
                error_text = "Hoards with missing location data not plotted (Id, No. of coins): " + "; ".join(error_details)
                st.text_area("Rows with Errors", value=error_text, height=200, disabled=True)
            else:
                st.text_area("Rows with Errors", value="no hoards with missing location data", height=200, disabled=True)
    except Exception as e:
        st.error(f"Error processing CSV file: {e}") 
