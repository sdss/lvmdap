
from os.path import isfile


from pyFIT3D.common.constants import __selected_extlaw__, __selected_R_V__, __n_Monte_Carlo__
from pyFIT3D.common.io import print_verbose
from pyFIT3D.common.auto_ssp_tools import ConfigAutoSSP
from pyFIT3D.common.gas_tools import ConfigEmissionModel

class ConfigRSP(ConfigAutoSSP):
    def __init__(self, config_file,
                 redshift_set=None, sigma_set=None, AV_set=None,
                 w_min=None, w_max=None, nl_w_min=None, nl_w_max=None,
                 mask_list=None, elines_mask_file=None,
                 sigma_inst=None, verbose=False, gas_fit=True):
        self.filename = config_file
        self.redshift_set = redshift_set
        self.sigma_set = sigma_set
        self.AV_set = AV_set
        self._verbose = verbose
        self.gas_fit = gas_fit
        self._load()

    def _load(self):
        """
        Loads the configuration file. Also, reads the configuration file of
        each to-be-fitted system.
        """
        config_keys = [
            'redshift', 'delta_redshift','min_redshift','max_redshift',
            'DV','RV','DS','RS','MIN_W','MAX_W',
            'sigma','delta_sigma','min_sigma','max_sigma',
            'AV','delta_AV','min_AV','max_AV',
        ]
        self.systems = []
        self.n_systems = 0
        # Array of `ConfigEmissionModel`
        self.systems_config = []
        self.start_w = 1e12
        self.end_w = -1e12

        if self.gas_fit:
            with open(self.filename, "r") as f:
                # get file size
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(0)

                # reads each line till systems config
                for k, v in zip(config_keys[0:10], f.readline().split()):
                    setattr(self, k, eval(v))
                for k, v in zip(config_keys[10:14], f.readline().split()):
                    setattr(self, k, eval(v))
                for k, v in zip(config_keys[14:18], f.readline().split()):
                    setattr(self, k, eval(v))
                self.n_systems = eval(f.readline())

                # reads each system config
                for i in range(self.n_systems):
                    l = f.readline().split()
                    tmp = {
                        'start_w': eval(l[0]), 'end_w': eval(l[1]), 'mask_file': l[2],
                        'config_file': l[3], 'npoly': eval(l[4]),
                        'mask_poly': l[5], 'nmin': eval(l[6]), 'nmax': eval(l[7])
                    }
                    if tmp['start_w'] < self.start_w:
                        self.start_w = tmp['start_w']
                    if tmp['end_w'] > self.end_w:
                        self.end_w = tmp['end_w']
                    if not isfile(tmp['mask_file']):
                        tmp['mask_file'] = None
                    self.systems.append(tmp)
                    self.systems_config.append(ConfigEmissionModel(tmp['config_file'], verbose=self._verbose))
                l = f.readline().split()
                self.MIN_DELTA_CHI_SQ = eval(l[0])
                self.MAX_N_ITER = eval(l[1])
                self.CUT_MEDIAN_FLUX = eval(l[2])
                self.ABS_MIN = 0.5*self.CUT_MEDIAN_FLUX

                l = f.readline().split()
                self.start_w_peak = eval(l[0])
                self.end_w_peak = eval(l[1])

                # Some configuration files could have this line
                # Not tested yet
                if (f.tell() != file_size):
                    l = f.readline().split()
                    if len(l) > 0:
                        self.wave_norm = eval(l[0])
                        self.w_wave_norm = eval(l[1])
                        self.new_ssp_file = l[2]

            print_verbose(f'{self.n_systems} Number of systems', verbose=self._verbose)
        
        # redefine parameters setted by user
        if self.redshift_set is not None:
            self.redshift = self.redshift if self.redshift_set[0] is None else self.redshift_set[0]
            self.delta_redshift = self.delta_redshift if self.redshift_set[1] is None else self.redshift_set[1]
            self.min_redshift = self.min_redshift if self.redshift_set[2] is None else self.redshift_set[2]
            self.max_redshift = self.max_redshift if self.redshift_set[3] is None else self.redshift_set[3]
        if self.sigma_set is not None:
            self.sigma = self.sigma if self.sigma_set[0] is None else self.sigma_set[0]
            self.delta_sigma = self.delta_sigma if self.sigma_set[1] is None else self.sigma_set[1]
            self.min_sigma = self.min_sigma if self.sigma_set[2] is None else self.sigma_set[2]
            self.max_sigma = self.max_sigma if self.sigma_set[3] is None else self.sigma_set[3]
        if self.AV_set is not None:
            self.AV = self.AV if self.AV_set[0] is None else self.AV_set[0]
            self.delta_AV = self.delta_AV if self.AV_set[1] is None else self.AV_set[1]
            self.min_AV = self.min_AV if self.AV_set[2] is None else self.AV_set[2]
            self.max_AV = self.max_AV if self.AV_set[3] is None else self.AV_set[3]
        
        self.MIN_DELTA_CHI_SQ = 0.0001
        self.MAX_N_ITER = 1
        self.CUT_MEDIAN_FLUX = 0.0
        self.ABS_MIN = 0.5*self.CUT_MEDIAN_FLUX

        return None