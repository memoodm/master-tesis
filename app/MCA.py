import prince

from datetime import datetime

class MCA:

    def getCoordinates(self,data):
        print(f"[MCA {datetime.now()}] START getCoordinates")
        mca = prince.MCA(
            n_components=data.shape[1],
            copy=True,
            check_input=True,
            engine='sklearn')     
        mca = mca.fit(data)
        coordinates_row =  mca.row_coordinates(data)
        coordinates_column = mca.column_coordinates(data)
        print(f"[MCA {datetime.now()}] END getCoordinates")
        return coordinates_row, coordinates_column