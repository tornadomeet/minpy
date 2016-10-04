from __future__ import absolute_import
from __future__ import print_function

import os
import yaml
from minpy.array_variants import ArrayType
from minpy.utils import log

# pylint: disable= invalid-name
_logger = log.get_logger(__name__)
# pylint: enable= invalid-name

# TODO: integrate this part into normal routine when MXNet fixes exception in
# Python.
mxnet_support_types = {'float', 'float16', 'float32', 'float64'}
mxnet_type_compatible_ops = {'negative', 'add', 'subtract', 'multiply',
                             'divide', 'true_divide', 'mod', 'power'}

class RuleError(ValueError):
    """Error in rule processing"""
    pass


class Rules(object):
    """Rules interface."""

    _rules = None
    _hash = None
    _env_var = '$MINPY_CONF'
    _conf_file = '.minpy_rules.conf'

    def __init__(self):
        self.load_rules_config()

    @classmethod
    def _build_hash(cls):
        """Clear hash and rebuild hash by rules"""
        raise NotImplementedError()

    @classmethod
    def load_rules_config(cls, force=False):
        """Load rules configuration from configs and build hash.
        
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
            locs = (os.curdir, os.path.expandvars(cls._env_var),
                    os.path.expanduser('~'))
            for loc in locs:
                try:
                    with open(os.path.join(loc, cls._conf_file)) as f:
                        config = yaml.safe_load(f)
                    break
                except IOError:
                    pass
                except yaml.YAMLError:
                    _logger.warn('Find corrupted configuration at {}'.format(loc))
            if config is None:
                _logger.error("Cannot find MinPy's rule configuration {} "
                                    "at {}. You can also use {} to specify.".format(
                                        cls._conf_file, locs, cls._env_var))
                config = {}
            else:
                _logger.debug('Use rule configuration at {}'.format(loc))
            cls._rules = config
            cls._build_hash()

    @classmethod
    def save_rules_config(cls):
        '''Save rules configuration from configs and build hash.

        Save
        '''
        with open(os.path.join(os.path.expandvars('~'), self._conf_file)) as f:
                  yaml.safe_dump(self._rules, f, default_flow_style=False)

    @classmethod
    def reset_rules(cls):
        """Reset rules.

        Delete all current rules. Also clear hash.
        """
        cls._rules = {}
        cls._hash = {}

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
        if impl_type != ArrayType.MXNET:
            return True
        if name in mxnet_type_compatible_ops:
            return True

        def is_supported_array_type(x):
            if isinstance(x, Array):
                # TODO: simplify here when MXNet, NumPy .dtype behavior become consistent
                return numpy.dtype(x.dtype).name in mxnet_support_types
            else:
                return True

        if not all(is_supported_array_type(x) for x in args):
            return False

        if name in self._hash and (self._hash[name] is None or
                                    self._get_arg_rule_key(args, kwargs) in
                                    self._hash[name]):
            return False
        return True
    
    def add(self, name, impl_type, args, kwargs):
        if impl_type != ArrayType.MXNET:
            raise RuleError('This rule only blacklists MXNet ops.')

        # Return type sequence
        type_seq = lambda args: [self._get_type_signiture(x) for x in args]

        self._rules.setdefault(name, [])
        self._rules[name].append({'args': type_seq(args), 'kwargs':
                                  list(kwargs.keys())})
        self._hash.setdefault(name, set())
        self._hash[name].add(self._get_arg_rule_key(args, kwargs))

    @classmethod
    def _build_hash(cls):
        cls._hash = {}
        for k, v in cls._rules.items():
            cls._hash[k] = set()
            for x in v:
                cls._hash[k].add('-'.join(x['args']) + '+' +
                                  '-'.join(sorted(x['kwargs'])))

    @staticmethod
    def _get_type_signiture(x):
        if isinstance(x, Array):
            return 'array_dim' + str(x.dim)
        elif isinstance(x, Number):
            return type(x.val).__name__
        else:
            return type(x).__name__

    @staticmethod
    def _get_arg_rule_key(args, kwargs):
        arg_key = [self.get_type_signiture(x) for x in args]
        kwarg_key = sorted(kwargs.keys())
        return '-'.join(arg_key) + '+' + '-'.join(kwarg_key)   
