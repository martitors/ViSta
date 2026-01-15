
# ============================================================
#  VISTA CLASS — FULL VERSION WITH ENGLISH DOCSTRINGS
# ============================================================

import os
import shutil
import numpy as np
from collections import Counter
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


def _assert_inside_casa():
    """
    Description:
        Check that the script is running inside a CASA environment.

    Inputs:
        None.

    Outputs:
        Ensures CASA modules are importable before proceeding.
    """
    try:
        import casatools
        import casatasks
    except ImportError:
        raise EnvironmentError(
            "\n[ViSta] ERROR: CASA modules not found.\n"
            "This code can ONLY run inside CASA.\n"
        )


_assert_inside_casa()


def _assert_all_exist(ms_list, func_name):
    """
    Description:
        Verify that all Measurement Set paths in a list exist on disk.

    Inputs:
        ms_list: List of Measurement Set paths to check.
        func_name: Name of the caller function, used for error messages.

    Outputs:
        Raises an error if any Measurement Set is missing; otherwise
        confirms that all inputs exist.
    """
    missing = [ms for ms in ms_list if not os.path.exists(ms)]
    if missing:
        raise FileNotFoundError(
            f"\n[ViSta:{func_name}] ERROR: These Measurement Sets do not exist:\n"
            + "\n".join(f"  - {ms}" for ms in missing)
            + "\nAborting.\n"
        )


class ViSta:
    """
    Description:
        Pipeline for rest-framing, centering, rebinning, stacking and
        averaging radio interferometric Measurement Sets using CASA.

    Inputs:
        input_file: Path to a text file listing MS paths and metadata.
        clean_previous: If True, delete intermediate MS products when
            they are no longer needed.

    Outputs:
        A ViSta instance with internal state initialized from the
        input file and ready to run the pipeline steps.
    """

    def __init__(self, input_file, clean_previous=False):
        """
        Description:
            Initialize ViSta, load CASA tools and parse the input file.

        Inputs:
            input_file: Path to the configuration file with MS and
                source information.
            clean_previous: If True, intermediate MS products are
                removed automatically.

        Outputs:
            Internal attributes are initialized, including ms_list,
            z_list, ra_list, dec_list, factors and ms_original.
        """
        self._load_casa()

        self.ms_list = []
        self.z_list = []
        self.ra_list = []
        self.dec_list = []
        self.factors = []
        self.original_phasecenters = []

        self.clean_previous = clean_previous

        self._load_input(input_file)
        self.ms_original = list(self.ms_list)

    def _load_casa(self):
        """
        Description:
            Load CASA tools required for table access and MS operations.

        Inputs:
            None.

        Outputs:
            Attributes tb, mstransform, phaseshift and concat are
            attached to this instance for later use.
        """
        from casatools import table as tbtool
        from casatasks import mstransform, phaseshift, concat

        self.tb = tbtool()
        self.mstransform = mstransform
        self.phaseshift = phaseshift
        self.concat = concat

        print("[ViSta] CASA modules successfully loaded.")

    def _load_input(self, path):
        """
        Description:
            Read and parse the input configuration file containing
            Measurement Sets and source metadata.

        Inputs:
            path: Path to the text file listing MS, redshift, RA, Dec
                and an optional scaling factor.

        Outputs:
            Populates ms_list, z_list, ra_list, dec_list and factors
            according to the contents of the input file.
        """
        rows = []
        temp_factors = []

        with open(path, "r") as f:
            for line in f:
                if line.strip() == "" or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 4:
                    raise ValueError("Each input row must contain at least: ms z ra dec")

                name = parts[0]
                z = float(parts[1])
                ra = parts[2]
                dec = parts[3]

                factor = None
                if len(parts) >= 5 and parts[4].strip() != "":
                    factor = float(parts[4])

                rows.append((name, z, ra, dec, factor))
                temp_factors.append(factor)

        if all(f is None for f in temp_factors):
            self.factors = []
            for name, z, ra, dec, _ in rows:
                self.ms_list.append(name)
                self.z_list.append(z)
                self.ra_list.append(ra)
                self.dec_list.append(dec)
            return

        for name, z, ra, dec, factor in rows:
            self.ms_list.append(name)
            self.z_list.append(z)
            self.ra_list.append(ra)
            self.dec_list.append(dec)
            self.factors.append(1.0 if factor is None else factor)

    def _safe_remove(self, path):
        """
        Description:
            Remove an intermediate Measurement Set directory if it is
            not one of the original inputs.

        Inputs:
            path: Filesystem path to a Measurement Set directory.

        Outputs:
            Deletes the directory at the given path when it is an
            intermediate product, leaving original MSs intact.
        """
        if os.path.exists(path) and path not in self.ms_original:
            shutil.rmtree(path)

    def _update_ms_list(self, suffix):
        """
        Description:
            Append a suffix to all current Measurement Set names in
            the internal list.

        Inputs:
            suffix: String to append to each entry of ms_list.

        Outputs:
            Updates ms_list so that each path has the given suffix
            appended.
        """
        self.ms_list = [f"{ms}{suffix}" for ms in self.ms_list]

    def restframing(self):
        """
        Description:
            Convert all current Measurement Sets to the rest frame
            using their redshifts.

        Inputs:
            None directly; uses ms_list and z_list stored in the
                instance.

        Outputs:
            Creates new Measurement Sets with suffix ".rest" where
            UVW, antenna positions and spectral frequencies have been
            scaled to the rest frame, and updates ms_list to point to
            the new files.
        """
        _assert_all_exist(self.ms_list, "restframing")

        old_list = list(self.ms_list)

        for vis, z in zip(old_list, self.z_list):
            out_file = f"{vis}.rest"

            print(f"\n[ViSta] → RESTFRAMING '{vis}'  (z = {z:.5f})")

            if os.path.exists(out_file):
                print(f"[ViSta]   WARNING: '{out_file}' already exists → skipping.")
                continue

            shutil.copytree(vis, out_file)

            self.tb.open(out_file, nomodify=False)
            self.tb.putcol("UVW", self.tb.getcol("UVW") / (1 + z))
            self.tb.flush()
            self.tb.close()
            

            self.tb.open(f"{out_file}/ANTENNA", nomodify=False)
            self.tb.putcol("POSITION", self.tb.getcol("POSITION") / (1 + z))
            self.tb.putcol("DISH_DIAMETER", self.tb.getcol("DISH_DIAMETER") / (1 + z))
            self.tb.flush()
            self.tb.close()
            

            self.tb.open(f"{out_file}/SPECTRAL_WINDOW", nomodify=False)
            cols = [
                "CHAN_FREQ", "REF_FREQUENCY", "CHAN_WIDTH",
                "EFFECTIVE_BW", "RESOLUTION", "TOTAL_BANDWIDTH"
            ]
            for col in cols:
                data = self.tb.getvarcol(col)
                for k in data:
                    data[k] *= (1 + z)
                self.tb.putvarcol(col, data)
            self.tb.flush()
            self.tb.close()
            

        self._update_ms_list(".rest")

        if self.clean_previous:
            for ms in old_list:
                self._safe_remove(ms)

        print("\n[ViSta]  Restframing completed.\n")

    def collect_original_phasecenters(self):
        """
        Description:
            Measure and store the original phase centers for all current
            Measurement Sets.

        Inputs:
            None directly; uses ms_list stored in the instance.

        Outputs:
            Fills original_phasecenters with SkyCoord objects
            describing the phase center of each MS.
        """
        self.original_phasecenters = [self.get_ms_phasecenter(ms) for ms in self.ms_list]

    def get_ms_phasecenter(self, ms_path, field_id=0, frame="icrs"):
        """
        Description:
            Read the phase center of a Measurement Set as a SkyCoord.

        Inputs:
            ms_path: Path to the Measurement Set to inspect.
            field_id: ID of the field from which to read PHASE_DIR.
            frame: Name of the coordinate frame for the returned
                SkyCoord (e.g. "icrs").

        Outputs:
            Returns a SkyCoord containing the phase center coordinates
            for the requested field in the specified frame.
        """
        _assert_all_exist([ms_path], "get_ms_phasecenter")
        self.tb.open(f"{ms_path}/FIELD")
        pd = self.tb.getcol("PHASE_DIR")
        self.tb.close()
        ra = np.degrees(pd[0][field_id][0]) / 15.0
        dec = np.degrees(pd[1][field_id][0])
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame=frame)

    def centering(self, rest_frequency=None):
        """
        Description:
            Shift the phase center of each Measurement Set to the RA/Dec
            specified in the input file and optionally set a rest frequency.

        Inputs:
            rest_frequency: Rest frequency in Hz to store in the SOURCE
                table of the centered Measurement Sets, or None to skip.

        Outputs:
            Creates new Measurement Sets with suffix ".center" where the
            phase center and pointing directions are updated, and updates
            ms_list to reference the centered files.
        """
        _assert_all_exist(self.ms_list, "centering")

        if not self.original_phasecenters:
            self.collect_original_phasecenters()

        old_list = list(self.ms_list)

        for i, vis in enumerate(old_list):
            out_file = f"{vis}.center"
            ra = self.ra_list[i]
            dec = self.dec_list[i].replace(":", ".")

            print(f"\n[ViSta] → CENTERING '{vis}' → new phase center RA={ra}, DEC={dec}")

            if os.path.exists(out_file):
                print(f"[ViSta]   WARNING: '{out_file}' exists → skipping.")
                continue

            self.phaseshift(
                vis=vis,
                outputvis=out_file,
                phasecenter=f"ICRS {ra} {dec}"
            )
            

            self.tb.open(f"{out_file}/FIELD", nomodify=False)
            for col in ["PHASE_DIR", "REFERENCE_DIR", "DELAY_DIR"]:
                v = self.tb.getvarcol(col)
                for k in v:
                    v[k] *= 0
                self.tb.putvarcol(col, v)
            self.tb.flush()
            self.tb.close()
            

            if rest_frequency is not None:
                self.tb.open(f"{out_file}/SOURCE", nomodify=False)
                x = self.tb.getcol("REST_FREQUENCY")
                x[0][0] = rest_frequency
                self.tb.putcol("REST_FREQUENCY", x)
                self.tb.flush()
                self.tb.close()
                

            self.tb.open(f"{out_file}/POINTING", nomodify=False)
            for col in ["DIRECTION", "TARGET"]:
                v = self.tb.getcol(col) * 0
                self.tb.putcol(col, v)
            self.tb.flush()
            self.tb.close()
            

        self._update_ms_list(".center")

        if self.clean_previous:
            for ms in old_list:
                self._safe_remove(ms)

        print("\n[ViSta]  Centering completed.\n")

    def rebinning(self, central_freq, spw=0, nn=None):
        """
        Description:
            Rebin all Measurement Sets to the same spectral resolution and
            same number of channels, centered around central_freq as closely
            as possible.
    
        Inputs:
            central_freq: Central frequency in Hz.
            spw: Spectral window index to use (default 0).
            nn: Number of final channels (if None, the smallest even number
                from all MSs is used).
    
        Outputs:
            Creates new Measurement Sets with suffix ".rebin", all rebinned
            to the same spectral resolution and channel count, and updates
            ms_list to reference the rebinned versions.
        """
    
        _assert_all_exist(self.ms_list, "rebinning")
        
        if central_freq is None:
    	    raise ValueError("central_freq is required for rebinning()")
       
        print("\n[ViSta] === STEP 1: Collecting spectral information from MS ===")
    
        spectral_res_list = []
        nchan_list = []
        freq_ranges = []
        old_list = list(self.ms_list)
    	
        for ms in old_list:
            spw_table = f"{ms}/SPECTRAL_WINDOW"
            self.tb.open(spw_table)
    
            chan_freqs = self.tb.getcol("CHAN_FREQ").T[0]
            chan_widths = self.tb.getcol("CHAN_WIDTH").T[0]
            self.tb.close()
    
            spectral_res = abs(chan_widths[0])
            spectral_res_list.append(spectral_res)
            nchan_list.append(len(chan_freqs))
            freq_ranges.append((chan_freqs.min(), chan_freqs.max()))
    
    
        spectral_res = max(spectral_res_list)
        print(f"[ViSta] Maximum spectral resolution among MS = {spectral_res:.2f} Hz")
    
        if nn is None:
            nn = min(nchan_list)
            if nn % 2 != 0:
                nn -= 1
        else:
            if nn > min(nchan_list):
                raise ValueError("nn is too large for the smallest MS.")
    
        print(f"[ViSta] Final number of channels (nn) = {nn}")
        print(f"[ViSta] Channel width (spectral_res) = {spectral_res:.2f} Hz")
    
        start_ideal = central_freq - spectral_res * nn / 2
        end_ideal   = central_freq + spectral_res * nn / 2
        print(f"[ViSta] Ideal start: {start_ideal:.3e}, ideal end: {end_ideal:.3e}\n")
    
        print("[ViSta] === STEP 2: Rebinning each MS ===")
    
        new_list = []
    
        for i, ms in enumerate(self.ms_list):
            msmin, msmax = freq_ranges[i]
            start, end = start_ideal, end_ideal
    
            print(f"\n[ViSta] --- Processing {ms} ---")
            print(f"[ViSta] Original MS range: ({msmin:.3e}, {msmax:.3e})")
            print(f"[ViSta] Start/End BEFORE shift: {start:.3e}, {end:.3e}")
    
            if start < msmin:
                nstep = int(np.ceil((msmin - start) / spectral_res))
                start += nstep * spectral_res
                end = start + nn * spectral_res
                
    
            if end > msmax:
                nstep = int(np.ceil((end - msmax) / spectral_res))
                start -= nstep * spectral_res
                end = start + nn * spectral_res
                
    
            print(f"[ViSta] Start/End FINAL:  {start:.3e}, {end:.3e}")
           
    
            out_file = f"{ms}.rebin"
    
            if os.path.exists(out_file):
                print(f"[ViSta] WARNING: {out_file} already exists! Skipping.\n")
                continue
    
            self.mstransform(
                vis=ms,
                outputvis=out_file,
                datacolumn="all",
                spw=str(spw),
                regridms=True,
                mode="frequency",
                nchan=nn,
                start=f"{start}Hz",
                width=f"{int(spectral_res)}Hz"
            )
    
            print(f"[ViSta] Rebinning completed → output: {out_file}\n")    
      
        self._update_ms_list(".rebin")
        self.big_res = nn

        if self.clean_previous:
            for ms in old_list:
                self._safe_remove(ms)
   
        print("[ViSta] ✔ Rebinning step completed.\n")


    def luminosity_scaling_factors(self, z_common, H0=70, Om0=0.3):
        """
        Description:
            Compute scaling factors that rescale source luminosities to
            a common redshift using a flat Lambda-CDM cosmology.

        Inputs:
            z_common: Redshift to which all sources will be rescaled.
            H0: Hubble constant in km/s/Mpc used in the cosmology.
            Om0: Matter density parameter used in the cosmology.

        Outputs:
            Returns a list of multiplicative factors, one per source in
            z_list, suitable for scaling flux densities to z_common.
        """
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        dl_common = cosmo.luminosity_distance(z_common).value
        dl_sources = [cosmo.luminosity_distance(z).value for z in self.z_list]

        return [
            (dl / dl_common) ** 2 * (1 + z_common) / (1 + z)
            for dl, z in zip(dl_sources, self.z_list)
        ]

    def coverage_factor(self):
        """
        Description:
            Compute a spectral coverage weight vector based on how many
            Measurement Sets overlap at each frequency channel.

        Inputs:
            None directly; uses ms_list and big_res set by rebinning.

        Outputs:
            Returns a 3D numpy array of shape (1, big_res, 1)
            containing weights for each spectral channel.
        """
        _assert_all_exist(self.ms_list, "coverage_factor")

        fmins, fmaxs = [], []

        for ms in self.ms_list:
            self.tb.open(f"{ms}/SPECTRAL_WINDOW")
            freqs = self.tb.getcol("CHAN_FREQ")[:, 0]
            self.tb.close()
            fmins.append(freqs.min())
            fmaxs.append(freqs.max())

        grid_min = min(fmins)
        grid_max = max(fmaxs)

        biggrid = np.linspace(grid_min, grid_max, self.big_res)
        Neff = np.zeros(self.big_res, dtype=int)

        for fmin, fmax in zip(fmins, fmaxs):
            mask = (biggrid >= fmin) & (biggrid <= fmax)
            Neff[mask] += 1

        w = np.sqrt(Neff / Neff.max())
        w[np.isnan(w)] = 0
        w[np.isinf(w)] = 0

        return w.reshape(1, len(w), 1)

    def weighting_singular(self, ms, factor, chunk=False, chunksize=10000):
        """
        Description:
            Multiply the DATA column of a single Measurement Set by a
            scalar or array factor, optionally in chunks.

        Inputs:
            ms: Path to the Measurement Set whose DATA will be scaled.
            factor: Scalar or array to multiply the DATA column by.
            chunk: If True, operate in row chunks to reduce memory use.
            chunksize: Number of rows per chunk when chunk is True.

        Outputs:
            Updates the DATA column of the specified Measurement Set
            to include the requested scaling.
        """
        _assert_all_exist([ms], "weighting_singular")

        self.tb.open(ms, nomodify=False)

        if not chunk:
            data = self.tb.getcol("DATA")
            data *= factor
            self.tb.putcol("DATA", data)
            self.tb.flush()
            self.tb.close()
            return

        nrows = self.tb.nrows()
        start = 0

        while start < nrows:
            nrow = min(chunksize, nrows - start)
            block = self.tb.getcol("DATA", startrow=start, nrow=nrow)
            block *= factor
            self.tb.putcol("DATA", block, startrow=start, nrow=nrow)
            start += nrow

        self.tb.flush()
        self.tb.close()

    def stacking(
        self,
        output_ms="stacked.ms",
        freqtolerance="1kHz",
        input_weighting=False,
        dl_weighting=False,
        pb_weighting=False,
        z_common=None,
        coverage_weighting=False,
        nu_pb=None,
        dish_diameter=None
    ):
        """
        Description:
            Concatenate all current Measurement Sets into a single MS
            with optional pre-weighting and coverage weighting.

        Inputs:
            output_ms: Path for the concatenated output Measurement Set.
            freqtolerance: Frequency tolerance string for CASA concat.
            input_weighting: If True, apply per-source factors from
                the input file before concatenation.
            dl_weighting: If True, apply luminosity-distance scaling
                factors to rescale all sources to z_common.
            pb_weighting: If True, apply primary-beam weighting using
                an external primary_beam_from_centers method.
            z_common: Common redshift used when dl_weighting is True.
            coverage_weighting: If True, apply coverage weights to the
                final concatenated MS after concat.
            nu_pb: Frequency in Hz at which to evaluate the primary
                beam for pb_weighting.
            dish_diameter: Dish diameter in meters used for primary
                beam calculations.

        Outputs:
            Writes a concatenated Measurement Set named output_ms and
            stores its path in concat_file for later use.
        """
        _assert_all_exist(self.ms_list, "stacking")

        print(f"\n[ViSta] → STACKING {len(self.ms_list)} Measurement Sets")

        if os.path.exists(output_ms):
            print(f"[ViSta] WARNING: '{output_ms}' already exists → skipping concatenation.")
            self.concat_file = output_ms
            return

        N = len(self.ms_list)
        final_pre_factors = [1.0] * N

        if input_weighting:
            if self.factors == []:
                raise ValueError("input_weighting=True but no factor in input file.")
            for i in range(N):
                final_pre_factors[i] *= self.factors[i]

        if dl_weighting:
            if z_common is None:
                raise ValueError("dl_weighting=True requires z_common.")
            lum_factors = self.luminosity_scaling_factors(z_common=z_common)
            for i in range(N):
                final_pre_factors[i] *= lum_factors[i]

        if pb_weighting:
            if nu_pb is None or dish_diameter is None:
                raise ValueError("pb_weighting=True requires nu_pb and dish_diameter.")
            for i, ms in enumerate(self.ms_list):
                theta, pb = self.primary_beam_from_centers(
                    index=i,
                    nu=nu_pb,
                    d=dish_diameter
                )
                if pb <= 0:
                    raise ValueError(f"[ViSta] PB value for MS '{ms}' is non-positive.")
                final_pre_factors[i] *= (1.0 / pb)

        for ms, w in zip(self.ms_list, final_pre_factors):
            if w != 1.0:
                print(f"[ViSta]   Applying pre-weighting factor {w:.4f} → {ms}")
                self.weighting_singular(ms, w)

        keys = [(ra.strip(), dec.strip())
                for ra, dec in zip(self.ra_list, self.dec_list)]
        counts = Counter(keys)

        visweightscale = []
        for key in keys:
            n = counts[key]
            visweightscale.append(1.0 / n if n > 0 else 1.0)

        print("[ViSta]   Concatenating Measurement Sets...")

        self.concat(
            vis=self.ms_list,
            freqtol=freqtolerance,
            concatvis=output_ms,
            visweightscale=visweightscale
        )

        if coverage_weighting:
            print("[ViSta]   Applying coverage weighting to the final MS...")
            cov = self.coverage_factor()
            if len(cov) == 0:
                raise RuntimeError("Coverage factor calculation returned no values.")
            self.weighting_singular(output_ms, cov)

        self.concat_file = output_ms

        print(f"[ViSta]  Stacking completed → saved as '{output_ms}'\n")

    def averaging(self, ms, output_ms=None):
        """
        Description:
            Time-average a Measurement Set after combining spectral
            windows and cleaning observation metadata.

        Inputs:
            ms: Path to the Measurement Set to time-average.
            output_ms: Path for the time-averaged output MS, or None to
                use a default name based on ms.

        Outputs:
            Creates a heavily time-averaged Measurement Set named
            output_ms (or ms + ".timeavg") and updates ms_list to
            contain only this product.
        """
        _assert_all_exist([ms], "averaging")

        print(f"\n[ViSta] → AVERAGING '{ms}'")

        self.tb.open(f"{ms}/SPECTRAL_WINDOW")
        nspw = self.tb.nrows()
        self.tb.close()

        if nspw > 1:
            combined = f"{ms}.spwcombined"
            if os.path.exists(combined):
                print(f"[ViSta]   WARNING: '{combined}' exists → skipping SPW combine.")
            else:
                print(f"[ViSta]   Combining all SPWs → '{combined}'")
                self.mstransform(vis=ms, outputvis=combined, datacolumn='all', combinespw=True)
            ms = combined

        self.tb.open(f"{ms}/OBSERVATION", nomodify=False)
        tr = self.tb.getcol("TIME_RANGE")
        tr[0, 0] = tr[0].min()
        tr[1, 0] = tr[1].max()
        self.tb.putcol("TIME_RANGE", tr)
        self.tb.flush()
        self.tb.close()

        print("[ViSta]   TIME_RANGE normalized.")

        self.tb.open(ms, nomodify=False)
        nrows = self.tb.nrows()
        if "SCAN_NUMBER" in self.tb.colnames():
            self.tb.putcol("SCAN_NUMBER", np.zeros(nrows, dtype=np.int32))
        if "OBSERVATION_ID" in self.tb.colnames():
            self.tb.putcol("OBSERVATION_ID", np.zeros(nrows, dtype=np.int32))
        self.tb.flush()
        self.tb.close()

        print("[ViSta]   MAIN table metadata cleaned.")

        if output_ms is None:
            output_ms = f"{ms}.timeavg"

        if os.path.exists(output_ms):
            print(f"[ViSta]   WARNING: '{output_ms}' exists → skipping averaging.")
            self.ms_list = [output_ms]
            return

        print(f"[ViSta]   Time-averaging MS → '{output_ms}'")

        self.mstransform(
            vis=ms,
            outputvis=output_ms,
            datacolumn='all',
            timeaverage=True,
            timebin="1e12s"
        )

        self.ms_list = [output_ms]

        print(f"[ViSta]  Averaging completed → saved as '{output_ms}'\n")
    
    def run(
        self,
        central_freq,
        weighting="none",
        imaging="tclean",
        output_image="vista_image"
    ):
        """
        Run the full ViSta pipeline:

            restframing →
            centering →
            rebinning (central_freq REQUIRED) →
            stacking (optional weighting) →
            imaging (tclean default, uvmultifit if chosen)

        Parameters
        ----------
        central_freq : float
            Mandatory. Central frequency for rebinning (Hz).
        weighting : str
            "none" (default), "input", "dl", "pb", "coverage".
        imaging : str
            "tclean" (default) or "uvmultifit".
        output_image : str
            Output name prefix for imaging.

        Behavior
        --------
        - tclean imaging: no averaging
        - uvmultifit: averaging is applied automatically
        """

        # -------------------------------
        # Check central frequency
        # -------------------------------
        if central_freq is None:
            raise ValueError("central_freq is REQUIRED for run().")

        print("\n========== ViSta RUN STARTED ==========\n")

        # -------------------------------
        # Step 1 — Restframing
        # -------------------------------
        self.restframing()

        # -------------------------------
        # Step 2 — Centering
        # -------------------------------
        self.centering()

        # -------------------------------
        # Step 3 — Rebinning
        # -------------------------------
        self.rebinning(central_freq=central_freq)

        # -------------------------------
        # Step 4 — Stacking
        # -------------------------------
        print(f"[RUN] Weighting selected: {weighting}")

        self.stacking(
            input_weighting      = (weighting == "input"),
            dl_weighting         = (weighting == "dl"),
            pb_weighting         = (weighting == "pb"),
            coverage_weighting   = (weighting == "coverage"),
            output_ms            = "stacked.ms"
        )

        final_ms = self.concat_file

        print(f"\n[RUN] Final MS after stacking: {final_ms}\n")

        # -------------------------------
        # Step 5 — Imaging
        # -------------------------------
        from imaging_methods import imaging_tclean, imaging_uvmultifit

        print(f"[RUN] Imaging method: {imaging}")

        if imaging == "tclean":
            # tclean does NOT require averaging
            imaging_tclean(final_ms, imagename=output_image)

        elif imaging == "uvmultifit":
            # uvmultifit MUST average the MS first
            avg_ms = final_ms + ".avg"
            self.averaging(final_ms, output_ms=avg_ms)

            imaging_uvmultifit(
                avg_ms,
                output=f"{output_image}.dat"
            )

        else:
            raise ValueError("Unknown imaging type: use 'tclean' or 'uvmultifit'.")

        print("\n========== ViSta RUN COMPLETE ==========\n")
        print(f"Output products begin with: {output_image}")

