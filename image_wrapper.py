from PIL import Image
from PIL.ExifTags import TAGS

import exifread as ef

class ImageWrapper(object):
    def __init__(self):
        pass

    def fetch_date(self, image: Image):
        
        dictionary = {}
        exifdata = image.getexif()

        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            if isinstance(data, bytes):
                data = data.decode()
            
            dictionary[tag] = data
        
        return dictionary["DateTime"] if "DateTime" in dictionary else ""


    def _convert_to_degress(self, value):
        """
        Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
        :param value:
        :type value: exifread.utils.Ratio
        :rtype: float
        """
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)

        return d + (m / 60.0) + (s / 3600.0)


    def fetch_latlong(self, filepath: str):
        '''
        returns gps data if present other wise returns empty dictionary
        '''
        with open(filepath, 'rb') as f:
            tags = ef.process_file(f)
            latitude = tags.get('GPS GPSLatitude')
            latitude_ref = tags.get('GPS GPSLatitudeRef')
            longitude = tags.get('GPS GPSLongitude')
            longitude_ref = tags.get('GPS GPSLongitudeRef')
            if latitude:
                lat_value = self._convert_to_degress(latitude)
                if latitude_ref.values != 'N':
                    lat_value = -lat_value
            else:
                return ""
            if longitude:
                lon_value = self._convert_to_degress(longitude)
                if longitude_ref.values != 'E':
                    lon_value = -lon_value
            else:
                return ""
            return f"{lat_value},{lon_value}"
        
        return ""
    
    def fetch_latlong_and_datetime(self, filepath: str):
        with open(filepath, 'rb') as f:
            tags = ef.process_file(f)
            datetimeoriginal = tags.get("EXIF DateTimeOriginal")
            latitude = tags.get('GPS GPSLatitude')
            latitude_ref = tags.get('GPS GPSLatitudeRef')
            longitude = tags.get('GPS GPSLongitude')
            longitude_ref = tags.get('GPS GPSLongitudeRef')
            if latitude:
                lat_value = self._convert_to_degress(latitude)
                if latitude_ref.values != 'N':
                    lat_value = -lat_value
            else:
                return {
                    "datetime": datetimeoriginal,
                    "latitude": None,
                    "longitude": None
                }
            if longitude:
                lon_value = self._convert_to_degress(longitude)
                if longitude_ref.values != 'E':
                    lon_value = -lon_value
            else:
                return {
                    "datetime": datetimeoriginal,
                    "latitude": None,
                    "longitude": None
                }
            return {
                    "datetime": datetimeoriginal,
                    "latitude": lat_value,
                    "longitude": lon_value
                }
        
        return {
            "datetime": datetimeoriginal,
            "latitude": None,
            "longitude": None
        }