import numpy as np
import numpy.random as npr
import pandas as pd
import scipy
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool

import os
import time
import itertools
from subprocess import STDOUT, PIPE, Popen

from cara import auction_parser
from cara import hyperparameters



class Auction:
    """Takes a Parser object and handles auction."""

    def __init__(self, parser: auction_parser.Parser):
        self.n = parser.n # num bidder classes
        self.m = parser.m # num items
        self.c = parser.c # num type classes
        self.r = parser.r # num valuation distributions
        self.t = parser.t # num discretization trials
        self.a = parser.a # num spa trials

        self.p_nc = parser.p_nc # bidder classes to type classes
        self.N_n  = parser.N_n  # bidder classes multiplicities
        self.d_mc = parser.d_mc # type classes to item valuations

        self.phi_r = parser.phi_r # valuation distributions
        self.inv_phi_r = parser.inv_phi_r # ppf of phi_r
        self.L_r   = parser.L_r   # lower end of support
        self.U_r   = parser.U_r   # upper end of support
        self.T_r   = parser.T_r   # type of distribution
        self.S_r   = parser.S_r   # size of discretization support
        self.P_r   = parser.P_r   # parameters for discretization type

        self.s      = None # discretized support size
        self.u_s    = None # current discretized support
        self.phi_rs = None # current discretized valuation distributions

        self.u_ts  = []  # discretized trial supports
        self.rev_t  = [] # discretized trial revenues
        self.time_t = [] # discretized trial times
        # i feel like this shouldn't be working properly cuz of how its handling communicating
        # between children workers but it seems like it is working so ima leave it how it is
        self.roasolver_time_t = [] # detailed timing from RoaSolver.jar
        
        self.bmps = self.unpack_bmps() # bidder multiplicity profiles

        self.discretized = False

        self.h = hyperparameters.DEFAULT_HYPERPARAMETERS
        # probably better way to get the right types
        h_ints = hyperparameters.H_INTS
        h_bools = hyperparameters.H_BOOLS
        h_strs = hyperparameters.H_STRS
        for k, v in parser.h.items():
            if k in h_ints:
                self.h[k] = int(v)
            elif k in h_bools:
                self.h[k] = bool(v)
            elif k in h_strs:
                self.h[k] = str(v)
            else:
                raise ValueError(f'Invalid hyperparameter "{k}"')


    def discretize(self) -> None:
        """Discretizes input according to parameters.
        """
        self.discretized = True

        dists_r = []
        for ri in range(self.r):
            masses, support = self.phi_r[ri].discretize(self.T_r[ri], int(self.S_r[ri]), self.P_r[ri])
            dists_r.append(pd.Series(masses, index=support, name=ri))
        df = pd.concat(dists_r, axis=1)
        df = df.sort_index()
        df = df.fillna(0.0)

        self.s = df.shape[0]
        self.u_s = df.index.values
        self.phi_rs = df.to_numpy().transpose()

        '''
        This version also works. This version gets support points from each valuation discretization
        and then assigns a stochastically-dominated mass to every point for every distribution.

        In the one above, instead, each distribution only has non zero masses on points that were 
        in its only discretization.

        I think the one above makes sense. Even though in some sense it is "wasting" support points
        when distributions have zero mass on them, I think the real runtime burden is not just the total
        size of the support, but the size of nonzero points in the support. Therefore there is little waste,
        and the version above is just more precise as each distribution uses its nonzero masses
        on presumably whatever support points are most valuable (if discretized well).

        supports = []
        for i in range(self.r):
            _, support = self.phi_r[i].discretize(self.T_r[i], int(self.S_r[i]), self.P_r[i])
            supports.append(support)
        supports = sorted(np.unique(np.array(supports).flatten()), reverse=True)
        
        dists_new = []
        for i in range(self.r):
            better_masses = self.phi_r[i].stochastically_dominated_masses(supports)
            dists_new.append(pd.Series(better_masses, index=supports, name=i))

        df_new = pd.concat(dists_new, axis=1)
        df_new = df_new.sort_index()
        df_new = df_new.fillna(0.0)

        self.s = df_new.shape[0]
        self.u_s = df_new.index.values
        self.phi_rs = df_new.to_numpy().transpose()
        '''

        
    def output(self) -> str:
        """Returns a properly-formatted string of the discretized parameters that
        serves as the input for the roasolver jar.
        """
        if not self.discretized:
            raise ValueError("Need to discretize auction before getting output.")

        s = f'{self.n} {self.m} {self.c} {self.r} {self.s}\n'

        # adding p_nc and N_n
        for ni in range(self.n):
            for ci in range(self.c):
                s += self._round_float(self.p_nc[ni][ci], self.h['type_class_sf']) + ' '
            s += self.N_n[ni]
            s += '\n'

        # adding d_mc
        for mi in range(self.m):
            for ci in range(self.c):
                s += self.d_mc[mi][ci]
                if ci < self.c - 1:
                    s += ' '
            s += '\n'

        # adding phi_rs
        for ri in range(self.r):
            for si in range(self.s):
                if si < self.s - 1:
                    s += self._round_float(self.phi_rs[ri][si], self.h['masses_sf'])
                    s += ' '
                else:
                    s += self._round_float(self.phi_rs[ri][si], self.h['final_mass_sf'], round_down=self.h['round_down_mass'])
            s += '\n'

        # adding u_s
        for si in range(self.s):
            if si < self.s - 1:
                s += self._round_float(self.u_s[si], self.h['support_sf'], round_down=self.h['round_down_support'])
                s += ' '
            else:
                s += self._round_float(self.u_s[si], self.h['final_support_sf'], round_down=self.h['round_down_support'])
        s += '\n'

        return s

    
    def run(self, args, output_file: str) -> None:
        """Runs concrete auctions self.a times and discretized trials self.t times.
        """
        self.args = args
        self.args.output = 'w' if not self.args.output else self.args.output

        # writing column headers for results and intermediate file.
        rev_cols = ['rev_'+'_'.join(x) for x in self.bmps]
        if self.t > 0:
            self.discretize()
            sup_cols = [f'sup_{i+1}' for i in range(self.s)]
            
            if self.args.full:
                with open(f'{output_file}.csv', self.args.output) as f:
                    f.write('trial,'+','.join(rev_cols+sup_cols) + '\n')
            if self.args.more:
                with open(f'{output_file}-int.txt', self.args.output) as f:
                    f.write('')

        # calculating concrete auction revenues
        if self.a > 0:
            spa_revs = self.run_spa()
            if self.args.full:
                spa_revs.to_csv(f'{output_file}-spa.csv', float_format="%.4f", mode=self.args.output)
        else:
            for _ in tqdm(range(0), desc='SPA', dynamic_ncols=True):
                pass
            spa_revs = pd.DataFrame()

        # doing all discretizations and storing them
        if not os.path.exists('temp'):
            os.makedirs('temp')
        intermediate_representations = []
        temp_files = []
        for ti in range(0, self.t):
            temp_file = f'temp/temp-{ti+1}.txt'
            
            self.discretize()
            self.u_ts.append(self.u_s)
            
            intermediate_representations.append(self.output())
            with open(temp_file, 'w') as f:
                f.write(self.output())
            if self.args.more:
                with open(f'{output_file}-int.txt', 'a') as f:
                    f.write(f"Trial {ti+1} intermediate input:\n")
                    f.write(self.output())
                    f.write('\n')
                
            temp_files.append(temp_file)

        # here is where we actually run the discretization trials
        def to_parallelize(temp_file, ti):
            st = time.time()
            cmd = ['java', '-jar', self.h['roasolver'], temp_file]
            proc = Popen(cmd, stdout=PIPE, stderr=STDOUT)
            stdout, stderr = proc.communicate(input='SomeInputstring')
            self._handle_roasolver_response(stdout, stderr, output_file, ti)
            et = time.time()
            return {'stdout': stdout, 'stderr': stderr, 'temp_file':temp_file, 'trial_time':et-st}

        pool = ProcessingPool()
        results_t = []
        for x in tqdm(pool.imap_unordered(to_parallelize, temp_files, range(self.t)), total=self.t, desc='Discretization', dynamic_ncols=True):
            results_t.append(x)

        # getting rid of temp files
        for file in os.listdir('temp'):
            os.remove(os.path.join('temp', file))
        
        # reformats the trials in correct order in output_file
        if self.args.full and self.t > 0:
            with open(f'{output_file}.csv', 'r+') as f:
                lines = f.readlines()
                f.seek(0)
                f.truncate()
                f.writelines(lines[:-self.t])
        
        # this we reprint in correct order if -full and add revs to revs_t regardless
        # CHECK IF WE'RE WRITING EVERYTHING TO REVS_T TWICE. WTF IS GOING ON.
        for ti in range(self.t):
            self._handle_roasolver_response(results_t[ti]['stdout'], results_t[ti]['stderr'], output_file, ti)          
        
        # writing timing data
        if self.args.timing and self.t > 0:
            trials = pd.Series([i for i in range(1, self.t+1)], name='trial')
            time_cols = ['time_'+'_'.join(x) for x in self.bmps]
            pd.DataFrame(self.roasolver_time_t, columns=time_cols, index=trials).to_csv(f'{output_file}-time-pre.csv', float_format="%.4f", mode=self.args.output)
            self.time_t = [x['trial_time'] for x in results_t]
            pd.Series(self.time_t, name='time', index=trials).to_csv(f'{output_file}-time-gen.csv', float_format="%.4f", mode=self.args.output)

        # writing best revs to stdout
        trial_revs = pd.DataFrame(self.rev_t, columns=rev_cols)
        all_revs = pd.concat([trial_revs, spa_revs.iloc[:2]]).astype(np.float64)
        best_revs = all_revs.max()
        
        ind_max_len = np.max([len(idx) for idx in best_revs.index.values])
        v_max_len   = np.max([len(str(v).split('.')[0]) for v in best_revs.values]) + 5
        for idx, v in zip(best_revs.index.values, best_revs.values):
            print(f"{idx:{ind_max_len}}  {v:>{v_max_len}.4f}")


    def _handle_roasolver_response(self, stdout, stderr, output_file, ti):
        """Handles RoasSolver's response. Directly writes to files if self.args.full.
        Appends timing and revenue data to instance variables.
        """
        nan_row = [np.nan for _ in range(len(self.bmps))]
        if stderr:
            print(f"\nTrial number {ti+1} has error:\n" + str(stderr) + "\n")
            if self.args.full:
                with open(f'{output_file}.csv', 'a') as f:
                    f.write(','.join([str(x) for x in nan_row]) + '\n')
            self.rev_t.append(nan_row)

        s = stdout.decode('utf-8')
        index = s.find('n_1') 
        df = pd.DataFrame(np.array(s[index:].split()).reshape(-1, 2+self.n))

        if df.empty:
            print(f"\nTrial number {ti+1} failed without error. Here is stdout:\n" + str(stdout) + "\n")
            if self.args.full:
                with open(f'{output_file}.csv', 'a') as f:
                    f.write(','.join([str(x) for x in nan_row]) + '\n')
            self.rev_t.append(nan_row)
        else:
            revs  = df.iloc[:,-2].values[1:]
            detailed_times = df.iloc[:,-1].values[1:]
            self.rev_t.append(revs)
            self.roasolver_time_t.append(detailed_times)
            if self.args.full:
                with open(f'{output_file}.csv', 'a') as f:
                        f.write(f'{ti+1},'+','.join(revs) + ',')

        if self.args.full:
            with open(f'{output_file}.csv', 'a') as f:
                f.write(','.join([f'{x:.4f}' for x in self.u_ts[ti]]) + '\n')
    

    def run_spa(self):
        """Does the entire spa revenue calcuations
        """
        # so we can use better sampling with self.inv_phi_r
        sample_phi_r = {}
        for i, inv_phi in enumerate(self.inv_phi_r):
            if inv_phi:
                def helper(i=i):
                    # this is because the eval needs a y variable
                    y = npr.sample() 
                    return eval(self.inv_phi_r[i])
                sample_phi_r[i] = helper
            else:
                sample_phi_r[i] = lambda i=i: self.phi_r[i].rvs()

        # takes array of valuation distributions and samples all
        def sample_phi(s):
            if s[0] == 'D':
                return sample_phi_r[int(s[1:])-1]()
            else:
                return int(s)
        sample_phi = np.vectorize(sample_phi)

        # only when >1 bidders
        bmps_sum = pd.Series([np.sum([int(bm) for bm in bmp]) for bmp in self.bmps])
        bmps_temp = pd.Series(self.bmps)[bmps_sum > 1]

        p_nc_normalized = self.p_nc / np.sum(self.p_nc, 1).reshape(-1, 1)

        sep_revs_list = []
        bun_revs_list = []
        pbar = tqdm(total=len(self.bmps)*self.a, desc='SPA', dynamic_ncols=True)
        for bmp in bmps_temp:
            bmp_ints = [int(b) for b in bmp] # maybe just change default type of bmp stuff to int instead of str
            bmp_sep_revs = []
            bmp_bun_revs = []
            for _ in range(self.a):
                valuations_dfs_list = []
                for i, bm in enumerate(bmp_ints):
                    bm_types = pd.Series(npr.choice(self.c, size=bm, p=p_nc_normalized[i]))

                    bm_valuation_dists = pd.DataFrame()
                    for i in range(self.m): # setting each column one-by-one --- prob nice pythonic way to explode this out instead
                        bm_valuation_dists[i] = bm_types.apply(lambda t: self.d_mc[i,t])

                    bm_valuations = sample_phi(bm_valuation_dists)
                    valuations_dfs_list.append(pd.DataFrame(bm_valuations))
                valuations = pd.concat(valuations_dfs_list)

                sep_rev = np.sum(np.sort(valuations, 0)[-2,:])
                bmp_sep_revs.append(sep_rev)
                bun_rev = np.sort(np.sum(valuations, 1), 0)[-2]
                bmp_bun_revs.append(bun_rev)
                pbar.update(1)

            sep_revs_list.append(pd.Series(bmp_sep_revs))
            bun_revs_list.append(pd.Series(bmp_bun_revs))
        pbar.close()

        rev_cols = ['rev_'+'_'.join(x) for x in bmps_temp]
        sep_revs_df = pd.DataFrame(dict(zip(rev_cols, sep_revs_list)))
        bun_revs_df = pd.DataFrame(dict(zip(rev_cols, bun_revs_list)))
        sep_revs = sep_revs_df.mean() # average rev from all trials is reported rev
        bun_revs = bun_revs_df.mean()
        sep_revs.name = 'rev_spa_separate'
        bun_revs.name = 'rev_spa_bundle'

        # computing confidence intervals
        # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
        sep_conf = sep_revs_df.apply(lambda a: scipy.stats.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=scipy.stats.sem(a)))
        bun_conf = bun_revs_df.apply(lambda a: scipy.stats.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=scipy.stats.sem(a)))
        sep_conf = sep_conf.T
        bun_conf = bun_conf.T
        sep_conf.columns = ['0.95_conf_low_separate', '0.95_conf_high_separate']
        bun_conf.columns = ['0.95_conf_low_bundle', '0.95_conf_high_bundle']

        spa_data = pd.concat([sep_revs, bun_revs, sep_conf, bun_conf], axis=1).T
        return spa_data
        
    
    def _round_float(self, x:float, sig_figs:int, round_down=False) -> str:
        '''Rounds float to a certain number of sig figs and returns string representation.
        '''
        # CHECK THAT THIS FUNCTION WORKS AS EXPECTED -- HAVE SOME REASONS TO THINK IT MIGHT BE OFF A BIT !!!

        str_rep = f'{x:.{sig_figs}g}'

        # if we rounded up but wanna round down
        if float(str_rep) > x and round_down:

            # depending on if str_rep is exponential form or not, need to find the smallest digit differently
            if 'e' in str_rep:
                if 'e-' in str_rep:
                    smallest_change = 1 / 10 ** (int(str_rep.split('e-')[-1]) + sig_figs - 1)
                else:
                    smallest_change = 1 / 10 ** (int(str_rep.split('e+')[-1]) + sig_figs - 1)
            else:
                smallest_change = 1 / 10 ** len(str_rep.split('.')[-1])

            x = float(str_rep) - smallest_change
            str_rep = f'{x:.{sig_figs}g}'
        
        return str(str_rep)

    def unpack_bms(self, s:str) -> list:
        """Unpacks a single bidder's multiplicities like '10;20~25;5'. Returns list of strings
        """
        strings = s.split(';')
        bidders = []
        for string in strings:
            if '~' in string:
                for i in range(int(string.split('~')[0]), int(string.split('~')[1])+1):
                    bidders.append(str(i))
            else:
                bidders.append(string)
        return bidders
    
    def unpack_bmps(self) -> list:
        """Unpacks all bidder multiplicity profiles. Returns list of tuples of strings.
        """
        a = [self.unpack_bms(s) for s in self.N_n]
        return list(itertools.product(*a))