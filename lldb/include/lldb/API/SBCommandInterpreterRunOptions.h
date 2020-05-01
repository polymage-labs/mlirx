//===-- SBCommandInterpreterRunOptions.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBCOMMANDINTERPRETERRUNOPTIONS_H
#define LLDB_API_SBCOMMANDINTERPRETERRUNOPTIONS_H

#include <memory>

#include "lldb/API/SBDefines.h"

namespace lldb {

class LLDB_API SBCommandInterpreterRunOptions {
  friend class SBDebugger;
  friend class SBCommandInterpreter;

public:
  SBCommandInterpreterRunOptions();
  ~SBCommandInterpreterRunOptions();

  bool GetStopOnContinue() const;

  void SetStopOnContinue(bool);

  bool GetStopOnError() const;

  void SetStopOnError(bool);

  bool GetStopOnCrash() const;

  void SetStopOnCrash(bool);

  bool GetEchoCommands() const;

  void SetEchoCommands(bool);

  bool GetEchoCommentCommands() const;

  void SetEchoCommentCommands(bool echo);

  bool GetPrintResults() const;

  void SetPrintResults(bool);

  bool GetAddToHistory() const;

  void SetAddToHistory(bool);

  bool GetAutoHandleEvents() const;

  void SetAutoHandleEvents(bool);

  bool GetSpawnThread() const;

  void SetSpawnThread(bool);

private:
  lldb_private::CommandInterpreterRunOptions *get() const;

  lldb_private::CommandInterpreterRunOptions &ref() const;

  // This is set in the constructor and will always be valid.
  mutable std::unique_ptr<lldb_private::CommandInterpreterRunOptions>
      m_opaque_up;
};

} // namespace lldb

#endif // LLDB_API_SBCOMMANDINTERPRETERRUNOPTIONS_H
