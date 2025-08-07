import geocoder

def get_location_by_ip():
    g = geocoder.ip('me')
    if g.ok:
        return {
            "city": g.city,
            "region": g.state,
            "country": g.country,
            "lat": g.latlng[0],
            "lng": g.latlng[1]
        }
    
    else:
        return {"error": "Could not fetch location."}
    
location = get_location_by_ip()
#print(location)
print(location["lat"], location["lng"])

