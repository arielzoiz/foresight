"""Run the statistics and draw every figure into out/."""
import fig_appendix_diff_hist
import fig_appendix_qq
import fig_changes_per_epoch
import fig_changes_per_epoch_all_rows
import fig_evoscore_by_run_length
import fig_evoscore_paired
import fig_progress_curve
import fig_progress_curve_all_rows
import stats_tests

if __name__ == "__main__":
    stats_tests.main()
    for fig in (fig_evoscore_paired, fig_progress_curve, fig_progress_curve_all_rows, fig_evoscore_by_run_length, fig_changes_per_epoch,
                fig_changes_per_epoch_all_rows,
                fig_appendix_diff_hist, fig_appendix_qq):
        fig.main()
