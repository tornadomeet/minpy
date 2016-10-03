from __future__ import absolute_import
from __future__ import print_function

import os
import yaml
from minpy.array_variants import ArrayType


class Rules(object):
    """Rules interface."""

    _rules = None

    def __init__(self):
        self._env_var = '$MINPY_CONF'
        self._conf_file = '.minpy_rules.conf'
        self.load_rules_config()

    @classmethod
    def load_rules_config(cls, force=False):
        """Load rules configuration from configs.
        
        Find rule configuration at current directory, self._env_var, and user's
        root in order. Then load the config into corresponding class variable.
        Load empty rules if loading fails.

        Parameters
        ----------
        force : bool
            if True, force to load configuration.
        """
        # TODO: add package data through installation
        # http://peak.telecommunity.com/DevCenter/setuptools#non-package-data-files
        if cls._rules is None or force:
            config = None
            locs = (os.curdir, os.path.expandvars(self._conf_file),
                    os.path.expanduser('~'))
            for loc in locs:
                try:
                    with open(os.path.join(loc, self._conf_file)) as f:
                        config = yaml.safe_load(f)
                    break
                except IOError:
                    pass
                except yaml.YAMLError:
                    _logger.warn('Find corrupted configuration at {}'.format(loc))
            if config is None:
                raise _logger.error("Cannot find MinPy's rule configuration {} "
                                    "at {}. You can also use {} to specify.".format(
                                        self._conf_file, locs, self._env_var))
                config = {}
            else:
                _logger.debug('Use rule configuration at {}'.format(loc))
            cls._rules = config

    @classmethod
    def reset_rules(cls):
        """Reset rules.

        Delete all current rules.
        """
        cls._rules = {}

    def allow(self, name, impl_type, args, kwargs):
        """Rule inquiry interface.

        Check if implementation is allowed.

        Parameters
        ----------
        name : str
            The dispatch name.
        impl_type : ArrayType 
            The type of implementation.
        args : list
            The positional arguments passed to the primitive.
        kwargs : dict
            The keyword arguments passed to the primitive.

        Returns
        -------
        bool
            True if implementation is allowed; False otherwize.
        """
        raise NotImplementedError()
    
    def add(self, name, impl_type, args, kwargs):
        """Rule registration interface.

        Register a new rule based on given info.

        Parameters
        ----------
        name : str
            The dispatch name.
        impl_type : ArrayType 
            The type of implementation.
        args : list
            The positional arguments passed to the primitive.
        kwargs : dict
            The keyword arguments passed to the primitive.
        """
        raise NotImplementedError()


class Blacklist(Rules):
    """Blacklist rules for rule-based policy"""

    def allow(self, name, impl_type, args, kwargs):
        pass
