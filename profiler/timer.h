/*****************************************************************************    
  TimerCpp : A C++ tool to measure times and extract statistics
  Copyright (C) 2014  Manos Tsardoulias

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*****************************************************************************/

#ifndef TIMERCPP_HPP
#define TIMERCPP_HPP

#include <iostream>
#include <cstdlib>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <sys/time.h>

//~ #include "defines.h"

class TimerCpp
{
  private:
    //~ Typedefs for internal use
    typedef std::map<std::string,double> Msd;
    typedef std::map<float,std::string> Mfs;
    typedef std::map<float,std::string>::iterator MfsIt;
    typedef std::map<std::string,unsigned long> Msul;
    typedef std::map<std::string,double>::iterator MsdIt;
    typedef std::map<std::string,double>::const_iterator MsdCIt;
    typedef std::pair<std::string,double> Psd;
    typedef std::pair<std::string,unsigned long> Psul;

    //~ Holds the time of each timer
    static Msd times;
    //~ Holds the max time of each timer
    static Msd max_time;
    //~ Holds the min time of each timer
    static Msd min_time;
    //~ Holds the mean time of each timer
    static Msd mean_time;
    //~ Holds the sum time of each timer
    static Msd sum_time;
    //~ Holds how many times each timer tic'ed
    static Msul count;
    //~ Internal variable to extract time
    static struct timeval msTime;
    
    //~ Structure capable of storing a "timer tree"
    static std::map<std::string,std::set<std::string> > timer_tree;
    
    //~ Holds the id of the tree root
    static std::string top_node;
    
    //~ Private constructor to avoid object creation. TimerCpp must be used
    //~ by static function calls
    TimerCpp(void){}

    //~ Internal functions to serve printing needs
    static void printMsInternal(double t);
    static void printSecInternal(double t);
    static void printMinInternal(double t);
    static void printHoursInternal(double t);
    static void printLiteralInternal(double t);
    
    //~ Used to print the time tree internally
    static void printIterativeTree(std::string node, std::string identation);
    //~ Used to print the mean time in a literal manner
    static void printLiteralMeanInternal(
      std::string timerId, 
      std::string identation = ""
    );

  public:

    //~ Starts a timer with "timerId" id. If you want to use the tree 
    //~ visualization, you must not provide a father id for the root.
    static void start(
      std::string timerId, 
      std::string father = ""
    );
      
    //~ Returns the time passed from the "start" call. Does not actually
    //~ stop the timer
    static double stop(std::string timerId);
    
    //~ Returns the mean time this timer has measured (only when "tic" is used)
    static double mean(std::string timerId);
    
    //~ Measures the time lapsed from start and stores min,max,mean values
    static void tic(std::string timerId);
    
    //~ Prints the milliseconds passed from "start". Internally calls "stop".
    static void printMs(std::string timerId);
    //~ Prints the seconds passed from "start". Internally calls "stop".
    static void printSec(std::string timerId);
    //~ Prints the minutes passed from "start". Internally calls "stop".
    static void printMin(std::string timerId);
    //~ Prints the hours passed from "start". Internally calls "stop".
    static void printHours(std::string timerId);
    
    //~ Prints the time passed from "start" in a literal manner. 
    //~ Internally calls "stop".
    static void printLiteral(std::string timerId);
    
    //~ Prints the mean time this timer has measured (only when "tic" is used)
    static void printLiteralMean(std::string timerId);
    
    //~ Prints the current value of all the timers in a literal manner
    static void printAll(void);
    //~ Prints the means of all timers in a literal fashion
    static void printAllMeans(void);
    //~ Prints the means of all timers in a tree-like structure
    static void printAllMeansTree(void);
    
};

#endif
