// This project contains source code from the Googletest framework
// obtained from https://github.com/google/googletest with the following
// terms:
//
// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
//
// Modifications and additions are published under the following terms:
//
// Copyright 2019, Simon Frasch
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------

#ifndef GUARD_GTEST_MPI_HPP
#define GUARD_GTEST_MPI_HPP
#include "../gtest.h"
#include <mpi.h>
#include <unistd.h>
#include <set>
#include <string>
#include "gtest_mpi_internal.hpp"

namespace gtest_mpi {
namespace { // no external linkage

class MPITestEnvironment : public ::testing::Environment {
public:
  MPITestEnvironment() : ::testing::Environment() {}

  MPITestEnvironment(const MPITestEnvironment&) = delete;

  MPITestEnvironment(MPITestEnvironment&&) = default;

  static MPI_Comm GetComm() { return global_test_comm; }

  void SetUp() override {
    if (global_test_comm != MPI_COMM_WORLD) {
      MPI_Comm_free(&global_test_comm);
      global_test_comm = MPI_COMM_WORLD;
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &global_test_comm);
  }

  void TearDown() override {
    if (global_test_comm != MPI_COMM_WORLD) {
      MPI_Comm_free(&global_test_comm);
      global_test_comm = MPI_COMM_WORLD;
    }
  }

private:
  static MPI_Comm global_test_comm;
};
MPI_Comm MPITestEnvironment::global_test_comm = MPI_COMM_WORLD;

class PrettyMPIUnitTestResultPrinter : public ::testing::TestEventListener {
public:
  PrettyMPIUnitTestResultPrinter()
      : rank_(0),
        comm_size_(1),
        comm_(MPITestEnvironment::GetComm()),
        num_sucessfull_tests_(0),
        num_failed_tests_(0) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &comm_size_);
  }

  // The following methods override what's in the TestEventListener class.
  void OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) override;
  void OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test) override;
  void OnTestCaseStart(const ::testing::TestCase& test_case) override;
  void OnTestStart(const ::testing::TestInfo& test_info) override;
  void OnTestPartResult(const ::testing::TestPartResult& result) override;
  void OnTestEnd(const ::testing::TestInfo& test_info) override;
  void OnTestCaseEnd(const ::testing::TestCase& test_case) override;
  void OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) override;
  void OnTestIterationEnd(const ::testing::UnitTest& unit_test, int iteration) override;

  void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& /*unit_test*/) override {}
  void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& /*unit_test*/) override {}
  void OnTestProgramStart(const ::testing::UnitTest& /*unit_test*/) override {}
  void OnTestProgramEnd(const ::testing::UnitTest& /*unit_test*/) override {}

private:
  int rank_;
  int comm_size_;
  MPI_Comm comm_;
  TestPartResultCollection failed_results_;
  int num_sucessfull_tests_;
  int num_failed_tests_;
  std::set<int> failed_ranks_;
  std::vector<TestInfoProperties> failed_test_properties_;
};

// Taken / modified from Googletest
void PrettyMPIUnitTestResultPrinter::OnTestIterationStart(const ::testing::UnitTest& unit_test,
                                                          int iteration) {
  using namespace ::testing;
  using namespace ::testing::internal;
  if (rank_ != 0) return;

  if (GTEST_FLAG(repeat) != 1)
    printf("\nRepeating all tests (iteration %d) . . .\n\n", iteration + 1);

  const char* const filter = GTEST_FLAG(filter).c_str();

  // Prints the filter if it's not *.  This reminds the user that some
  // tests may be skipped.
  if (!String::CStringEquals(filter, kUniversalFilter)) {
    ColoredPrintf(COLOR_YELLOW, "Note: %s filter = %s\n", GTEST_NAME_, filter);
  }

  if (ShouldShard(kTestTotalShards, kTestShardIndex, false)) {
    const Int32 shard_index = Int32FromEnvOrDie(kTestShardIndex, -1);
    ColoredPrintf(COLOR_YELLOW, "Note: This is test shard %d of %s.\n",
                  static_cast<int>(shard_index) + 1, internal::posix::GetEnv(kTestTotalShards));
  }

  if (GTEST_FLAG(shuffle)) {
    ColoredPrintf(COLOR_YELLOW, "Note: Randomizing tests' orders with a seed of %d .\n",
                  unit_test.random_seed());
  }

  ColoredPrintf(COLOR_GREEN, "[==========] ");
  printf("Running %s from %s.\n", FormatTestCount(unit_test.test_to_run_count()).c_str(),
         FormatTestCaseCount(unit_test.test_case_to_run_count()).c_str());
  fflush(stdout);
}

// Taken / modified from Googletest
void PrettyMPIUnitTestResultPrinter::OnEnvironmentsSetUpStart(
    const ::testing::UnitTest& /*unit_test*/) {
  ColoredPrintf(COLOR_GREEN, "[----------] ");
  printf("Global test environment set-up.\n");
  fflush(stdout);
}

// Taken / modified from Googletest
void PrettyMPIUnitTestResultPrinter::OnTestCaseStart(const ::testing::TestCase& test_case) {
  using namespace ::testing;
  using namespace ::testing::internal;
  if (rank_ != 0) return;
  const std::string counts = FormatCountableNoun(test_case.test_to_run_count(), "test", "tests");
  ColoredPrintf(COLOR_GREEN, "[----------] ");
  printf("%s from %s", counts.c_str(), test_case.name());
  if (test_case.type_param() == NULL) {
    printf("\n");
  } else {
    printf(", where %s = %s\n", kTypeParamLabel, test_case.type_param());
  }
  fflush(stdout);
}

// Taken / modified from Googletest
void PrettyMPIUnitTestResultPrinter::OnTestStart(const ::testing::TestInfo& test_info) {
  if (rank_ != 0) return;
  ColoredPrintf(COLOR_GREEN, "[ RUN      ] ");
  printf("%s.%s", test_info.test_case_name(), test_info.name());

  printf("\n");
  fflush(stdout);
}

// Taken / modified from Googletest
void PrettyMPIUnitTestResultPrinter::OnTestPartResult(const ::testing::TestPartResult& result) {
  using namespace ::testing;
  using namespace ::testing::internal;
  // If the test part succeeded, we don't need to do anything.
  if (result.type() == TestPartResult::kSuccess) return;
  failed_results_.Add(result);
}

// Taken / modified from Googletest
void PrintFailedTestResultCollection(const TestPartResultCollection& collection, int rank) {
  for (std::size_t i = 0; i < collection.Size(); ++i) {
    std::string m =
        (::testing::Message() << "Rank " << rank << ": "
                              << ::testing::internal::FormatFileLocation(
                                     collection.file_names.get_str(i), collection.line_numbers[i])
                              << " "
                              << TestPartResultTypeToString(
                                     ::testing::TestPartResult::Type(collection.types[i]))
                              << collection.messages.get_str(i))
            .GetString();
    printf("%s\n", m.c_str());
    fflush(stdout);
  }
}

// Taken / modified from Googletest
void PrettyMPIUnitTestResultPrinter::OnTestEnd(const ::testing::TestInfo& test_info) {
  using namespace ::testing;
  using namespace ::testing::internal;

  // check if any ranks failed
  int failed_locally = failed_results_.Size() > 0;
  std::vector<int> failed_flags_per_rank;
  if (rank_ == 0) failed_flags_per_rank.resize(comm_size_);
  MPI_Gather(&failed_locally, 1, MPI_INT, failed_flags_per_rank.data(), 1, MPI_INT, 0, comm_);

  // failed non-root ranks Send to root and exit
  if (rank_ != 0) {
    if (failed_locally) {
      failed_results_.Send(comm_, 0);
    }
    failed_results_.Reset();
    return;
  }

  int failed_globally = failed_locally;
  for (const auto& f : failed_flags_per_rank) {
    if (f) failed_globally = 1;
  }

  // print root failure fist
  if (failed_results_.Size() > 0) {
    PrintFailedTestResultCollection(failed_results_, rank_);
    failed_ranks_.insert(0);
  }

  // receive and print from other failed ranks
  for (int r = 1; r < comm_size_; ++r) {
    if (failed_flags_per_rank[r]) {
      failed_ranks_.insert(r);
      failed_results_.Recv(comm_, r);
      PrintFailedTestResultCollection(failed_results_, r);
    }
  }

  // Reset result storage before next test
  failed_results_.Reset();

  if (!failed_globally) {
    ColoredPrintf(COLOR_GREEN, "[       OK ] ");
    ++num_sucessfull_tests_;
  } else {
    ColoredPrintf(COLOR_RED, "[  FAILED  ] ");
    ++num_failed_tests_;
    TestInfoProperties prop;
    if (test_info.name()) prop.name = test_info.name();
    if (test_info.test_case_name()) prop.case_name = test_info.test_case_name();
    if (test_info.should_run()) prop.should_run = test_info.should_run();
    if (test_info.type_param()) prop.type_param = test_info.type_param();
    if (test_info.value_param()) prop.value_param = test_info.value_param();
    for (int r = 0; r < comm_size_; ++r) {
      if (failed_flags_per_rank[r]) {
        prop.ranks.insert(r);
      }
    }
    failed_test_properties_.emplace_back(std::move(prop));
  }

  printf("%s.%s", test_info.test_case_name(), test_info.name());
  if (failed_globally) PrintFullTestCommentIfPresent(test_info);

  if (GTEST_FLAG(print_time)) {
    printf(" (%s ms)\n", internal::StreamableToString(test_info.result()->elapsed_time()).c_str());
  } else {
    printf("\n");
  }
  fflush(stdout);
}

// Taken / modified from Googletest
void PrettyMPIUnitTestResultPrinter::OnTestCaseEnd(const ::testing::TestCase& test_case) {
  using namespace ::testing;
  if (!GTEST_FLAG(print_time) || rank_ != 0) return;

  const std::string counts = FormatCountableNoun(test_case.test_to_run_count(), "test", "tests");
  ColoredPrintf(COLOR_GREEN, "[----------] ");
  printf("%s from %s (%s ms total)\n\n", counts.c_str(), test_case.name(),
         internal::StreamableToString(test_case.elapsed_time()).c_str());
  fflush(stdout);
}

static std::string FormatSet(const std::set<int>& s) {
  std::string res;
  for (const auto& val : s) {
    res += std::to_string(val);
    if (val != *(--s.end())) {
      res += ", ";
    }
  }
  // res.resize(res.size() - 2); // remove last comma
  return res;
}

// Taken / modified from Googletest
static void PrintFullTestCommentIfPresent(const std::string& type_param,
                                          const std::string& value_param) {
  if (!type_param.empty() || !value_param.empty()) {
    printf(", where ");
    if (!type_param.empty()) {
      printf("%s = %s", kTypeParamLabel, type_param.c_str());
      if (!value_param.empty()) printf(" and ");
    }
    if (!value_param.empty()) {
      printf("%s = %s", kValueParamLabel, value_param.c_str());
    }
    printf(",");
  }
}

// Taken / modified from Googletest
void PrettyMPIUnitTestResultPrinter::OnTestIterationEnd(const ::testing::UnitTest& unit_test,
                                                        int /*iteration*/) {
  using namespace ::testing;
  failed_results_.Reset();
  if (rank_ != 0) {
    return;
  }

  ColoredPrintf(COLOR_GREEN, "[==========] ");
  printf("%s from %s ran on %d ranks.", FormatTestCount(unit_test.test_to_run_count()).c_str(),
         FormatTestCaseCount(unit_test.test_case_to_run_count()).c_str(), comm_size_);
  if (GTEST_FLAG(print_time)) {
    printf(" (%s ms total)", internal::StreamableToString(unit_test.elapsed_time()).c_str());
  }
  printf("\n");
  ColoredPrintf(COLOR_GREEN, "[  PASSED  ] ");
  printf("%s.\n", FormatTestCount(num_sucessfull_tests_).c_str());

  if (num_failed_tests_) {
    ColoredPrintf(COLOR_RED, "[  FAILED  ] ");
    printf("%s, listed below:\n", FormatTestCount(num_failed_tests_).c_str());
    for (const auto& prop : failed_test_properties_) {
      if (!prop.should_run) continue;
      ColoredPrintf(COLOR_RED, "[  FAILED  ] ");
      printf("%s.%s", prop.case_name.c_str(), prop.name.c_str());
      PrintFullTestCommentIfPresent(prop.type_param, prop.value_param);
      printf(" on ranks [%s]", FormatSet(prop.ranks).c_str());
      printf("\n");
    }
  }

  int num_disabled = unit_test.reportable_disabled_test_count();
  if (num_disabled && !GTEST_FLAG(also_run_disabled_tests)) {
    if (!num_failed_tests_) {
      printf("\n"); // Add a spacer if no FAILURE banner is displayed.
    }
    ColoredPrintf(COLOR_YELLOW, "  YOU HAVE %d DISABLED %s\n\n", num_disabled,
                  num_disabled == 1 ? "TEST" : "TESTS");
  }
  // Ensure that Google Test output is printed before, e.g., heapchecker output.
  fflush(stdout);
}

// Taken / modified from Googletest
void PrettyMPIUnitTestResultPrinter::OnEnvironmentsTearDownStart(
    const ::testing::UnitTest& /*unit_test*/) {
  if (rank_ != 0) return;
  ColoredPrintf(COLOR_GREEN, "[----------] ");
  printf("Global MPI test environment tear-down\n");
  fflush(stdout);
}

} // anonymous namespace
} // namespace gtest_mpi

#endif

