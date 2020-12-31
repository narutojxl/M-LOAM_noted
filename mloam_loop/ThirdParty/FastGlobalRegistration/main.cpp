// ----------------------------------------------------------------------------
// -                       Fast Global Registration                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) Intel Corporation 2016
// Qianyi Zhou <Qianyi.Zhou@gmail.com>
// Jaesik Park <syncle@gmail.com>
// Vladlen Koltun <vkoltun@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
#include <stdio.h>
#include "app.h"

#define DIV_FACTOR 1.4		 // Division factor used for graduated non-convexity
#define USE_ABSOLUTE_SCALE 0 // Measure distance in absolute scale (1) or in scale relative to the diameter of the model (0)
#define MAX_CORR_DIST 0.025	 // Maximum correspondence distance (also see comment of USE_ABSOLUTE_SCALE)
#define ITERATION_NUMBER 64	 // Maximum number of iteration
#define TUPLE_SCALE 0.95	 // Similarity measure used for tuples of feature points.
#define TUPLE_MAX_CNT 1000	 // Maximum tuple numbers.

int main(int argc, char *argv[])
{
	if (argc != 4)
	{
        printf("Usage ::\n");
		printf("%s [feature_01] [feature_02] [transform_output_txt]\n", argv[0]);
		return 0;
	}
	// fgr::CApp app;
	fgr::CApp app(DIV_FACTOR, USE_ABSOLUTE_SCALE, MAX_CORR_DIST, ITERATION_NUMBER, TUPLE_SCALE, TUPLE_MAX_CNT);
	app.ReadFeature(argv[1]);
	app.ReadFeature(argv[2]);
	app.NormalizePoints();
	app.AdvancedMatching();
	app.OptimizePairwise(true);
	app.WriteTrans(argv[3]);
	return 0;
}
