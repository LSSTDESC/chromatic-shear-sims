import argparse
import pyarrow.compute as pc
import pyarrow.dataset as ds

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantiles",
        type=float,
        required=False,
        nargs='+',
        default=[0.50],
        help="Quantiles to compute"
    )
    return parser.parse_args()


def main():
    args = get_args()

    columns = [
      "mag_true_g_lsst",
      "mag_true_i_lsst",
    ]

    predicate = (pc.field("mag_true_r_lsst") < pc.scalar(26)) & (pc.field("mag_true_i_lsst") < pc.scalar(26)) & (pc.field("mag_true_z_lsst") < pc.scalar(26))

    dataset = ds.dataset("/oak/stanford/orgs/kipac/users/smau/cosmoDC2_v1.1.4_parquet")
    table = dataset.to_table(columns=columns, filter=predicate)
    # quantiles = [0.0, 1./3., 2./3., 1.0]  # tertiles
    # quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]  # quartiles
    # quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # quintiles
    quantiles = args.quantiles

    print(f"Calculating median")
    print(pc.approximate_median(pc.subtract(table["mag_true_g_lsst"], table["mag_true_i_lsst"])))

    print(f"Calculating quantiles: {quantiles}")
    print(pc.quantile(pc.subtract(table["mag_true_g_lsst"], table["mag_true_i_lsst"]), quantiles))

if __name__ == "__main__":
    main()
