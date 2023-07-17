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
    quantiles = args.quantiles

    print(f"Gals")
    print(f"Calculating median")
    print(pc.approximate_median(pc.subtract(table["mag_true_g_lsst"], table["mag_true_i_lsst"])))

    print(f"Calculating quantiles: {quantiles}")
    print(pc.quantile(pc.subtract(table["mag_true_g_lsst"], table["mag_true_i_lsst"]), quantiles))

    columns = [
      "gmag",
      "imag",
    ]
    predicate = (
        (pc.field("imag") > pc.scalar(19))
        & (pc.field("imag") < pc.scalar(21))
        & pc.match_substring_regex(pc.field("sedFilename"), "^k[mp]*")
    )
    dataset = ds.dataset("/oak/stanford/orgs/kipac/users/smau/dc2_stellar_healpixel.arrow", format="arrow")
    table = dataset.to_table(columns=columns, filter=predicate)

    print(f"stars")
    print(f"Calculating median")
    print(pc.approximate_median(pc.subtract(table["gmag"], table["imag"])))

    print(f"Calculating quantiles: {quantiles}")
    print(pc.quantile(pc.subtract(table["gmag"], table["imag"]), quantiles))

if __name__ == "__main__":
    main()
