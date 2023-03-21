import collections
import numpy as np
import os
import json
import h5py
import xarray as xr
#------------------------------------
# @ USER DEFINED IMPORTS START
# @ USER DEFINED IMPORTS END
#------------------------------------

class channel_psd():

    """Data model representing spectral density mesured by sensor
    """

    #------------
    # Constructor
    #------------
    def __init__(self,name=None):
#------------------------------------
# @ USER DEFINED DESCRIPTION START
# @ USER DEFINED DESCRIPTION END
#------------------------------------

        #------------------------------------------------------------
        # Optional parameters are set to None by default.
        # User shall manually create adequate constructor for the
        # optional properties before using them in the code.
        #------------------------------------------------------------
        # If a property is None or if it includes any None,
        # the full property is not exported with "pro_rep",
        # i.e saved in .json or .hdf5 files.
        #------------------------------------------------------------
        self._values=np.zeros(shape=(1,1), dtype=float)
        self._unit='[m**2/Hz]'
        self._min_value=0
        self._max_value=100
        self._short_name='channel_psd'
        self._description='Values of the power spectral density from movements measured by a specific sensor'
        self._name='none'
        if not(name == None): # pragma: no cover
            self._name = name

#------------------------------------
# @ USER DEFINED PROPERTIES START
# @ USER DEFINED PROPERTIES END
#------------------------------------

#------------------------------------
# @ USER DEFINED METHODS START
# @ USER DEFINED METHODS END
#------------------------------------

    #------------
    # Get functions
    #------------
    @ property
    def values(self): # pragma: no cover
        """:obj:`.numpy.ndarray` of :obj:`float`:power spectral density with the following index : (time, frequency) dim(*,*) []
        """
        return self._values
    #------------
    @ property
    def unit(self): # pragma: no cover
        """str: Power Spectral Density in m**2 / Hz
        """
        return self._unit
    #------------
    @ property
    def min_value(self): # pragma: no cover
        """float: Arbitrary minimum value of frequency and power spectral density . []
        """
        return self._min_value
    #------------
    @ property
    def max_value(self): # pragma: no cover
        """float: Arbitrary maximum value of frequency and power spectral density. []
        """
        return self._max_value
    #------------
    @ property
    def short_name(self): # pragma: no cover
        """str: none
        """
        return self._short_name
    #------------
    @ property
    def description(self): # pragma: no cover
        """str: Metadata : Data description
        """
        return self._description
    #------------
    @ property
    def name(self): # pragma: no cover
        """str: name of the instance object
        """
        return self._name
    #------------
    #------------
    # Set functions
    #------------
    @ values.setter
    def values(self,val): # pragma: no cover
        self._values=val
    #------------
    @ unit.setter
    def unit(self,val): # pragma: no cover
        self._unit=str(val)
    #------------
    @ min_value.setter
    def min_value(self,val): # pragma: no cover
        self._min_value=float(val)
    #------------
    @ max_value.setter
    def max_value(self,val): # pragma: no cover
        self._max_value=float(val)
    #------------
    @ short_name.setter
    def short_name(self,val): # pragma: no cover
        self._short_name=str(val)
    #------------
    @ description.setter
    def description(self,val): # pragma: no cover
        self._description=str(val)
    #------------
    @ name.setter
    def name(self,val): # pragma: no cover
        self._name=str(val)
    #------------
    #-------------------------
    # Representation functions
    #-------------------------
    def type_rep(self): # pragma: no cover
        """Generate a representation of the object type

        Returns:
            :obj:`collections.OrderedDict`: dictionnary that contains the representation of the object type
        """

        rep = collections.OrderedDict()
        rep["__type__"] = "generic_sensors:channel_psd"
        rep["name"] = self.name
        rep["description"] = self.description
        return rep
    def prop_rep(self, short = False, deep = True):

        """Generate a representation of the object properties

        Args:
            short (:obj:`bool`,optional): if True, properties are represented by their type only. If False, the values of the properties are included.
            deep (:obj:`bool`,optional): if True, the properties of each property will be included.

        Returns:
            :obj:`collections.OrderedDict`: dictionnary that contains a representation of the object properties
        """

        rep = collections.OrderedDict()
        rep["__type__"] = "generic_sensors:channel_psd"
        rep["name"] = self.name
        rep["description"] = self.description
        if self.is_set("values"):
            if (short):
                rep["values"] = str(self.values.shape)
            else:
                try:
                    rep["values"] = self.values.astype(float).round(2).tolist()
                except:
                    rep["values"] = self.values
        if self.is_set("unit"):
            rep["unit"] = self.unit
        if self.is_set("min_value"):
            rep["min_value"] = self.min_value
        if self.is_set("max_value"):
            rep["max_value"] = self.max_value
        if self.is_set("short_name"):
            rep["short_name"] = self.short_name
        if self.is_set("description"):
            rep["description"] = self.description
        if self.is_set("name"):
            rep["name"] = self.name
        return rep
    #-------------------------
    # Save functions
    #-------------------------
    def json_rep(self, short=False, deep=True):

        """Generate a JSON representation of the object properties

        Args:
            short (:obj:`bool`,optional): if True, properties are represented by their type only. If False, the values of the properties are included.
            deep (:obj:`bool`,optional): if True, the properties of each property will be included.

        Returns:
            :obj:`str`: string that contains a JSON representation of the object properties
        """

        return ( json.dumps(self.prop_rep(short=short, deep=deep),indent=4, separators=(",",": ")))
    def saveJSON(self, fileName=None):
        """Save the instance of the object to JSON format file

        Args:
            fileName (:obj:`str`, optional): Name of the JSON file, included extension. Defaults is None. If None, the name of the JSON file will be self.name.json. It can also contain an absolute or relative path.

        """

        if fileName==None:
            fileName=self.name + ".json"
        f=open(fileName, "w")
        f.write(self.json_rep())
        f.write("\n")
        f.close()

    def saveHDF5(self,filePath=None):
        """Save the instance of the object to HDF5 format file
        Args:
            filePath (:obj:`str`, optional): Name of the HDF5 file, included extension. Defaults is None. If None, the name of the HDF5 file will be "self.name".h5. It can also contain an absolute or relative path.
        """
        # Define filepath
        if (filePath == None):
            if hasattr(self, "name"):
                filePath = self.name + ".h5"
            else:
                raise Exception("name is required for saving")
        # Open hdf5 file
        h = h5py.File(filePath,"w")
        group = h.create_group(self.name)
        # Save the object to the group
        self.saveToHDF5Handle(group)
        # Close file
        h.close()
        pass
    def saveToHDF5Handle(self, handle):
        """Save the properties of the object to the hdf5 handle.
        Args:
            handle (:obj:`h5py.Group`): Handle used to store the object properties
        """
        if self.is_set("values") :
            handle["values"] = np.array(self.values,dtype=float)
        if self.is_set("unit") :
            ar = []
            ar.append(self.unit.encode("ascii"))
            handle["unit"] = np.asarray(ar)
        if self.is_set("min_value") :
            handle["min_value"] = np.array([self.min_value],dtype=float)
        if self.is_set("max_value") :
            handle["max_value"] = np.array([self.max_value],dtype=float)
        if self.is_set("short_name") :
            ar = []
            ar.append(self.short_name.encode("ascii"))
            handle["short_name"] = np.asarray(ar)
        if self.is_set("description") :
            ar = []
            ar.append(self.description.encode("ascii"))
            handle["description"] = np.asarray(ar)
        if self.is_set("name") :
            ar = []
            ar.append(self.name.encode("ascii"))
            handle["name"] = np.asarray(ar)
    def create_xr_DataArray(self, user_dims= "time"):
        d = {
            "attrs":{
                "unit" : self.unit,
                "min_value" : self.min_value,
                "max_value" : self.max_value,
                "description" : self.description},
            "dims" : user_dims,
            "data" : self.values,
            "name" : self.short_name
        }
        xr_da = xr.DataArray.from_dict(d)
        return xr_da

    #-------------------------
    # Load functions
    #-------------------------
    def loadJSON(self,name = None, filePath = None):
        """Load an instance of the object from JSON format file

        Args:
            name (:obj:`str`, optional): Name of the object to load.
            filePath (:obj:`str`, optional): Path of the JSON file to load. If None, the function looks for a file with name "name".json.

        The JSON file must contain a datastructure representing an instance of this object's class, as generated by e.g. the function :func:`~saveJSON`.

        """

        if not(name == None):
            self.name = name
        if (name == None) and not(filePath == None):
            self.name = ".".join(filePath.split(os.path.sep)[-1].split(".")[0:-1])
        if (filePath == None):
            if hasattr(self, "name"):
                filePath = self.name + ".json"
            else:
                raise Exception("object needs name for loading.")
        if not(os.path.isfile(filePath)):
            raise Exception("file %s not found."%filePath)
        self._loadedItems = []
        f = open(filePath,"r")
        data = f.read()
        f.close()
        dd = json.loads(data)
        self.loadFromJSONDict(dd)
    def loadFromJSONDict(self, data):
        """Load an instance of the object from a dictionnary representing a JSON structure)

        Args:
            name (:obj:`collections.OrderedDict`): Dictionnary containing a JSON structure, as produced by e.g. :func:`json.loads`

        """

        varName = "values"
        try :
            setattr(self,varName, np.array(data[varName]))
        except :
            pass
        varName = "unit"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
        varName = "min_value"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
        varName = "max_value"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
        varName = "short_name"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
        varName = "description"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
        varName = "name"
        try :
            setattr(self,varName, data[varName])
        except :
            pass
    def loadHDF5(self,name=None, filePath=None):
        """Load an instance of the object from HDF5 format file
        Args:
            name (:obj:`str`, optional): Name of the object to load. Default is None.
            filePath (:obj:`str`, optional): Path of the HDF5 file to load. If None, the function looks for a file with name "self.name".h5.
        The HDF5 file must contain a datastructure representing an instance of this objects class, as generated by e.g. the function :func:`~saveHDF5`.
        """
        # Define filepath and name
        if not(name == None):
            self.name = name
        if (filePath == None):
            if hasattr(self, "name"):
                filePath = self.name + ".h5"
            else:
                raise Exception("name is required for loading")
        # Open hdf5 file
        h = h5py.File(filePath,"r")
        group = h[self.name]
        # Save the object to the group
        self.loadFromHDF5Handle(group)
        # Close file
        h.close()
        pass
    def loadFromHDF5Handle(self, gr):
        """Load the properties of the object from a hdf5 handle.
        Args:
            gr (:obj:`h5py.Group`): Handle used to read the object properties from
        """
        if ("values" in list(gr.keys())):
            self.values = gr["values"][:][:]
        if ("unit" in list(gr.keys())):
            self.unit = gr["unit"][0].decode("ascii")
        if ("min_value" in list(gr.keys())):
            self.min_value = gr["min_value"][0]
        if ("max_value" in list(gr.keys())):
            self.max_value = gr["max_value"][0]
        if ("short_name" in list(gr.keys())):
            self.short_name = gr["short_name"][0].decode("ascii")
        if ("description" in list(gr.keys())):
            self.description = gr["description"][0].decode("ascii")
        if ("name" in list(gr.keys())):
            self.name = gr["name"][0].decode("ascii")
    def loadFromXRDataArray(self, data):
        """Load an instance of the object from a Xarray.Dataarray

        Args:
            name (:obj:`Xarray.Dataarray`): N-dimensional array with labeled coordinates and dimensions, as produced by e.g. :func:`create_xr_DataArray`

        """

        if isinstance(data, xr.DataArray) :
            varName = "values"
            try :
                setattr(self,varName, np.array(data.data))
                # test is_set_function compatibility 
                len(getattr(self,varName))
            except TypeError as e :
                if str(e) == 'len() of unsized object':
                    setattr(self,varName, np.array([data.data]))
                else : pass
            except :
                pass
            varName = "unit"
            try :
                setattr(self,varName, data.unit)
            except :
                pass
            varName = "min_value"
            try :
                setattr(self,varName, data.min_value)
            except :
                pass
            varName = "max_value"
            try :
                setattr(self,varName, data.max_value)
            except :
                pass
            varName = "short_name"
            try :
                setattr(self,varName, data.short_name)
            except :
                pass
            varName = "description"
            try :
                setattr(self,varName, data.description)
            except :
                pass
            varName = "name"
            try :
                setattr(self,varName, data.name)
            except :
                pass
    #------------------------
    # is_set function
    #------------------------
    def is_set(self, varName): # pragma: no cover

        """Check if a given property of the object is set

        Args:
            varName (:obj:`str`): name of the property to check

        Returns:
            :obj:`bool`: True if the property is set, else False
        """

        if (isinstance(getattr(self,varName),list) ):
            if (len(getattr(self,varName)) > 0 and not any([np.any(a==None) for a in getattr(self,varName)])  ):
                return True
            else :
                return False
        if (isinstance(getattr(self,varName),np.ndarray) ):
            if (len(getattr(self,varName)) > 0 and not any([np.any(a==None) for a in getattr(self,varName)])  ):
                return True
            else :
                return False
        if (getattr(self,varName) != None):
            return True
        return False
    #------------------------
