import requests
import numpy as np
from scipy.stats import gaussian_kde
import pickle

def get_all_shifts(residue: str, atom_id: str) -> list[float]:
    """Get all chemical shifts for a residue/atom combination."""
    url = "https://api.bmrb.io/v2/search/chemical_shifts"
    params = {
        "comp_id": residue,
        "atom_id": atom_id,
        "database": "macromolecules"
    }
    
    response = requests.get(url, params=params, headers={"Application": "NMR-ShiftPredictor 1.0"})
    if response.status_code != 200:
        return []
    
    data = response.json()
    if not data.get('data'):
        return []
    
    shift_idx = data['columns'].index('Atom_chem_shift.Val')
    return [row[shift_idx] for row in data['data'] if row[shift_idx] is not None]


def get_atoms_for_residue(residue: str) -> list[str]:
    """Get all atom names observed for a residue."""
    # Query with just the residue to see what atoms exist
    url = "https://api.bmrb.io/v2/search/chemical_shifts"
    params = {
        "comp_id": residue,
        "database": "macromolecules"
    }
    
    response = requests.get(url, params=params, headers={"Application": "NMR-ShiftPredictor 1.0"})
    data = response.json()
    
    atom_idx = data['columns'].index('Atom_chem_shift.Atom_ID')
    atoms = set(row[atom_idx] for row in data['data'])
    return sorted(atoms)


class ShiftDistributions:
    """Store empirical chemical shift distributions for all residue/atom combinations."""
    
    RESIDUES = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL"
    ]
    
    def __init__(self):
        self.distributions = {}  # {(residue, atom): np.array of shifts}
        self.kdes = {}  # {(residue, atom): gaussian_kde object}
    
    def fetch_all(self, min_observations: int = 50):
        """Fetch all distributions from BMRB."""
        for res in self.RESIDUES:
            print(f"Fetching {res}...", flush=True)
            atoms = get_atoms_for_residue(res)
            
            for atom in atoms:
                shifts = get_all_shifts(res, atom)
                if len(shifts) >= min_observations:
                    key = (res, atom)
                    self.distributions[key] = np.array(shifts)
                    # Pre-compute KDE for fast likelihood evaluation
                    self.kdes[key] = gaussian_kde(shifts)
                    print(f"  {atom}: {len(shifts)} observations", flush=True)
    
    def likelihood(self, residue: str, atom: str, shift: float) -> float:
        """Get probability density at a given chemical shift."""
        key = (residue, atom)
        if key not in self.kdes:
            return 0.0
        return float(self.kdes[key](shift)[0])
    
    def log_likelihood(self, residue: str, atom: str, shift: float) -> float:
        """Get log probability density (more numerically stable)."""
        key = (residue, atom)
        if key not in self.kdes:
            return -np.inf
        return float(self.kdes[key].logpdf(shift)[0])
    
    def get_range(self, residue: str, atom: str, percentile: float = 99) -> tuple[float, float]:
        """Get the expected range for a residue/atom combination."""
        key = (residue, atom)
        if key not in self.distributions:
            return (np.nan, np.nan)
        data = self.distributions[key]
        low = np.percentile(data, (100 - percentile) / 2)
        high = np.percentile(data, 100 - (100 - percentile) / 2)
        return (low, high)
    
    def save(self, path: str):
        """Save distributions to disk (KDEs will be regenerated on load)."""
        with open(path, 'wb') as f:
            pickle.dump(self.distributions, f)
    
    def load(self, path: str):
        """Load distributions and regenerate KDEs."""
        with open(path, 'rb') as f:
            self.distributions = pickle.load(f)
        # Regenerate KDEs
        self.kdes = {key: gaussian_kde(vals) for key, vals in self.distributions.items()}


# Build and save
if __name__ == "__main__":
    dist = ShiftDistributions()
    dist.fetch_all()
    dist.save("bmrb_shift_distributions.pkl")
    
    # Example usage for your density crystallization
    print("\nExample likelihoods for a peak at 55.2 ppm:")
    for res in ["ALA", "VAL", "LEU", "THR"]:
        ll = dist.log_likelihood(res, "CA", 55.2)
        print(f"  {res} CA: {ll:.2f}")