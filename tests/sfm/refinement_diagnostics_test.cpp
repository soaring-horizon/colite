#include <gtest/gtest.h>
#include "colite/sfm/refinement_diagnostics.h"

namespace colite {
TEST(RefinementDiagnosticsTest, RecordsPrunedObservationsDuringCompletion) {
  using namespace refinement_internal;
  const Snapshot before = {{{1, 0}, {1, 1.0}}, {{2, 0}, {1, 2.0}}};
  const Snapshot after = {{{1, 0}, {1, 1.0}}};
  std::set<ObservationKey> removed;
  GlobalRefinementDiagnostics row;
  std::vector<GlobalRefinementObservationChange> events;
  Record(before, before, after, after, &removed, &row, &events);
  EXPECT_EQ(row.completion_removed, 1U);
  EXPECT_EQ(row.unique_removed, 1U);
  EXPECT_EQ(row.net_membership_changes, 1U);
  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(events.front().reason, "merge_pruning");
  EXPECT_EQ(removed.count({2, 0}), 1U);
}

TEST(RefinementDiagnosticsTest, SeparatesChurnFromNetMembershipAndReassignment) {
  using namespace refinement_internal;
  const Snapshot before = {{{1, 0}, {1, 1.0}}, {{2, 0}, {1, 2.0}}};
  const Snapshot ba = {{{1, 0}, {1, .5}}, {{2, 0}, {1, 1.0}}};
  const Snapshot completed = {{{1, 0}, {2, .5}}, {{2, 0}, {1, 1.0}},
                              {{3, 0}, {2, 5.0}}};
  const Snapshot filtered = {{{1, 0}, {2, .5}}, {{2, 0}, {1, 1.0}}};
  std::set<ObservationKey> removed = {{3, 0}};
  GlobalRefinementDiagnostics row;
  row.iteration = 2;
  std::vector<GlobalRefinementObservationChange> events;
  Record(before, ba, completed, filtered, &removed, &row, &events);
  EXPECT_EQ(row.unique_added, 1U);
  EXPECT_EQ(row.unique_removed, 1U);
  EXPECT_EQ(row.reassigned, 1U);
  EXPECT_EQ(row.readded_previous, 1U);
  EXPECT_EQ(row.added_then_removed, 1U);
  EXPECT_EQ(row.net_membership_changes, 0U);
  EXPECT_NEAR(row.common_p95_before, 1.95, 1e-12);
  EXPECT_NEAR(row.common_p95_after, .975, 1e-12);
  ASSERT_EQ(events.size(), 3U);
  EXPECT_EQ(events.back().reason, "reprojection");
}
}  // namespace colite
