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

#ifndef GUARD_GTEST_MPI_INTERNAL_HPP
#define GUARD_GTEST_MPI_INTERNAL_HPP
#include "../gtest.h"
#include <mpi.h>
#include <unistd.h>
#include <string>

namespace gtest_mpi {
namespace { // no external linkage

// Taken / modified from Googletest
static const char kDisableTestFilter[] = "DISABLED_*:*/DISABLED_*";
static const char kDeathTestCaseFilter[] = "*DeathTest:*DeathTest/*";
static const char kUniversalFilter[] = "*";
static const char kDefaultOutputFormat[] = "xml";
static const char kDefaultOutputFile[] = "test_detail";
static const char kTestShardIndex[] = "GTEST_SHARD_INDEX";
static const char kTestTotalShards[] = "GTEST_TOTAL_SHARDS";
static const char kTestShardStatusFile[] = "GTEST_SHARD_STATUS_FILE";
static const char kTypeParamLabel[] = "TypeParam";
static const char kValueParamLabel[] = "GetParam()";

// Taken / modified from Googletest
enum GTestColor { COLOR_DEFAULT, COLOR_RED, COLOR_GREEN, COLOR_YELLOW };

// Taken / modified from Googletest
static void PrintFullTestCommentIfPresent(const ::testing::TestInfo& test_info) {
  const char* const type_param = test_info.type_param();
  const char* const value_param = test_info.value_param();

  if (type_param != NULL || value_param != NULL) {
    printf(", where ");
    if (type_param != NULL) {
      printf("%s = %s", kTypeParamLabel, type_param);
      if (value_param != NULL) printf(" and ");
    }
    if (value_param != NULL) {
      printf("%s = %s", kValueParamLabel, value_param);
    }
  }
}

// Taken / modified from Googletest
bool ShouldUseColor(bool stdout_is_tty) {
  using namespace ::testing;
  using namespace ::testing::internal;
  const char* const gtest_color = GTEST_FLAG(color).c_str();

  if (String::CaseInsensitiveCStringEquals(gtest_color, "auto")) {
#if GTEST_OS_WINDOWS && !GTEST_OS_WINDOWS_MINGW
    // On Windows the TERM variable is usually not set, but the
    // console there does support colors.
    return stdout_is_tty;
#else
    // On non-Windows platforms, we rely on the TERM variable.
    const char* const term = getenv("TERM");
    const bool term_supports_color =
        String::CStringEquals(term, "xterm") || String::CStringEquals(term, "xterm-color") ||
        String::CStringEquals(term, "xterm-256color") || String::CStringEquals(term, "screen") ||
        String::CStringEquals(term, "screen-256color") || String::CStringEquals(term, "tmux") ||
        String::CStringEquals(term, "tmux-256color") ||
        String::CStringEquals(term, "rxvt-unicode") ||
        String::CStringEquals(term, "rxvt-unicode-256color") ||
        String::CStringEquals(term, "linux") || String::CStringEquals(term, "cygwin");
    return stdout_is_tty && term_supports_color;
#endif // GTEST_OS_WINDOWS
  }

  return String::CaseInsensitiveCStringEquals(gtest_color, "yes") ||
         String::CaseInsensitiveCStringEquals(gtest_color, "true") ||
         String::CaseInsensitiveCStringEquals(gtest_color, "t") ||
         String::CStringEquals(gtest_color, "1");
  // We take "yes", "true", "t", and "1" as meaning "yes".  If the
  // value is neither one of these nor "auto", we treat it as "no" to
  // be conservative.
}

// Taken / modified from Googletest
static const char* GetAnsiColorCode(GTestColor color) {
  switch (color) {
    case COLOR_RED:
      return "1";
    case COLOR_GREEN:
      return "2";
    case COLOR_YELLOW:
      return "3";
    default:
      return NULL;
  };
}

// Taken / modified from Googletest
static void ColoredPrintf(GTestColor color, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);

  static const bool in_color_mode = ShouldUseColor(isatty(fileno(stdout)) != 0);
  const bool use_color = in_color_mode && (color != COLOR_DEFAULT);

  if (!use_color) {
    vprintf(fmt, args);
    va_end(args);
    return;
  }

  printf("\033[0;3%sm", GetAnsiColorCode(color));
  vprintf(fmt, args);
  printf("\033[m"); // Resets the terminal to default.
  va_end(args);
}

// Taken / modified from Googletest
::testing::internal::Int32 Int32FromEnvOrDie(const char* var,
                                             ::testing::internal::Int32 default_val) {
  using namespace ::testing;
  using namespace ::testing::internal;
  const char* str_val = getenv(var);
  if (str_val == NULL) {
    return default_val;
  }

  Int32 result;
  if (!ParseInt32(Message() << "The value of environment variable " << var, str_val, &result)) {
    exit(EXIT_FAILURE);
  }
  return result;
}

// Taken / modified from Googletest
static std::string FormatCountableNoun(int count, const char* singular_form,
                                       const char* plural_form) {
  using namespace ::testing;
  return internal::StreamableToString(count) + " " + (count == 1 ? singular_form : plural_form);
}

// Taken / modified from Googletest
static std::string FormatTestCount(int test_count) {
  return FormatCountableNoun(test_count, "test", "tests");
}

// Taken / modified from Googletest
static std::string FormatTestCaseCount(int test_case_count) {
  return FormatCountableNoun(test_case_count, "test case", "test cases");
}

// Taken / modified from Googletest
static const char* TestPartResultTypeToString(::testing::TestPartResult::Type type) {
  switch (type) {
    case ::testing::TestPartResult::kSuccess:
      return "Success";

    case ::testing::TestPartResult::kNonFatalFailure:
    case ::testing::TestPartResult::kFatalFailure:
#ifdef _MSC_VER
      return "error: ";
#else
      return "Failure\n";
#endif
    default:
      return "Unknown result type";
  }
}

// Taken / modified from Googletest
bool ShouldShard(const char* total_shards_env, const char* shard_index_env,
                 bool in_subprocess_for_death_test) {
  using namespace ::testing;
  using namespace ::testing::internal;
  if (in_subprocess_for_death_test) {
    return false;
  }

  const Int32 total_shards = Int32FromEnvOrDie(total_shards_env, -1);
  const Int32 shard_index = Int32FromEnvOrDie(shard_index_env, -1);

  if (total_shards == -1 && shard_index == -1) {
    return false;
  } else if (total_shards == -1 && shard_index != -1) {
    const Message msg = Message() << "Invalid environment variables: you have " << kTestShardIndex
                                  << " = " << shard_index << ", but have left " << kTestTotalShards
                                  << " unset.\n";
    ColoredPrintf(COLOR_RED, msg.GetString().c_str());
    fflush(stdout);
    exit(EXIT_FAILURE);
  } else if (total_shards != -1 && shard_index == -1) {
    const Message msg = Message() << "Invalid environment variables: you have " << kTestTotalShards
                                  << " = " << total_shards << ", but have left " << kTestShardIndex
                                  << " unset.\n";
    ColoredPrintf(COLOR_RED, msg.GetString().c_str());
    fflush(stdout);
    exit(EXIT_FAILURE);
  } else if (shard_index < 0 || shard_index >= total_shards) {
    const Message msg =
        Message() << "Invalid environment variables: we require 0 <= " << kTestShardIndex << " < "
                  << kTestTotalShards << ", but you have " << kTestShardIndex << "=" << shard_index
                  << ", " << kTestTotalShards << "=" << total_shards << ".\n";
    ColoredPrintf(COLOR_RED, msg.GetString().c_str());
    fflush(stdout);
    exit(EXIT_FAILURE);
  }

  return total_shards > 1;
}

// info from TestInfo, which does not have a copy constructor
struct TestInfoProperties {
  std::string name;
  std::string case_name;
  std::string type_param;
  std::string value_param;
  bool should_run;
  std::set<int> ranks;
};

// Holds null terminated strings in a single vector,
// which can be exchanged in a single MPI call
class StringCollection {
public:
  void Add(const char* s) {
    int size = 0;
    for (; *s != '\0'; ++s, ++size) {
      text.push_back(*s);
    }
    text.push_back('\0');
    start_indices.push_back(prev_size);
    prev_size = size + 1;
  }

  // Sends content to requested rank
  void Send(MPI_Comm comm, int rank) const {
    MPI_Send(text.data(), text.size(), MPI_CHAR, rank, 0, comm);
    MPI_Send(start_indices.data(), start_indices.size(), MPI_INT, rank, 0, comm);
  }

  // Overrides content with data from requested rank
  void Recv(MPI_Comm comm, int rank) {
    MPI_Status status;
    int count = 0;

    // Recv text
    MPI_Probe(rank, 0, comm, &status);
    MPI_Get_count(&status, MPI_CHAR, &count);
    text.resize(count);
    MPI_Recv(text.data(), count, MPI_CHAR, rank, 0, comm, MPI_STATUS_IGNORE);

    // Recv sizes
    MPI_Probe(rank, 0, comm, &status);
    MPI_Get_count(&status, MPI_INT, &count);
    start_indices.resize(count);
    MPI_Recv(start_indices.data(), count, MPI_INT, rank, 0, comm, MPI_STATUS_IGNORE);
  }

  void Reset() {
    text.clear();
    start_indices.clear();
    prev_size = 0;
  }

  const char* get_str(const int id) const { return text.data() + start_indices[id]; }

  const std::size_t Size() const { return start_indices.size(); }

private:
  int prev_size = 0;
  std::vector<char> text;
  std::vector<int> start_indices;
};

// All info recuired to print a failed test result.
// Includes functionality for MPI exchange
struct TestPartResultCollection {
  // Sends content to requested rank
  void Send(MPI_Comm comm, int rank) {
    MPI_Send(types.data(), types.size(), MPI_INT, rank, 0, comm);
    MPI_Send(line_numbers.data(), line_numbers.size(), MPI_INT, rank, 0, comm);
    summaries.Send(comm, rank);
    messages.Send(comm, rank);
    file_names.Send(comm, rank);
  }

  // Overrides content with data from requested rank
  void Recv(MPI_Comm comm, int rank) {
    MPI_Status status;
    int count = 0;

    // Recv text
    MPI_Probe(rank, 0, comm, &status);
    MPI_Get_count(&status, MPI_INT, &count);
    types.resize(count);
    MPI_Recv(types.data(), count, MPI_INT, rank, 0, comm, MPI_STATUS_IGNORE);

    // Recv sizes
    MPI_Probe(rank, 0, comm, &status);
    MPI_Get_count(&status, MPI_INT, &count);
    line_numbers.resize(count);
    MPI_Recv(line_numbers.data(), count, MPI_INT, rank, 0, comm, MPI_STATUS_IGNORE);

    summaries.Recv(comm, rank);
    messages.Recv(comm, rank);
    file_names.Recv(comm, rank);
  }

  void Add(const ::testing::TestPartResult& result) {
    types.push_back(result.type());
    line_numbers.push_back(result.line_number());
    summaries.Add(result.summary());
    messages.Add(result.message());
    file_names.Add(result.file_name());
  }

  void Reset() {
    types.clear();
    line_numbers.clear();
    summaries.Reset();
    messages.Reset();
    file_names.Reset();
  }

  std::size_t Size() const { return types.size(); }

  std::vector<int> types;
  std::vector<int> line_numbers;
  StringCollection summaries;
  StringCollection messages;
  StringCollection file_names;
};

} // anonymous namespace
} // namespace gtest_mpi
#endif

