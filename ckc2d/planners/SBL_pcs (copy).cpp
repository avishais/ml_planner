/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Ioan Sucan */

//#include "ompl/geometric/planners/sbl/SBL.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/tools/config/SelfConfig.h"
#include <limits>
#include <cassert>

#include "SBL_pcs.h"

ompl::geometric::SBL::SBL(const base::SpaceInformationPtr &si, int joints_num, int passive_chains_num, double custom_num) : base::Planner(si, "SBL"), StateValidityChecker(si, joints_num, passive_chains_num, 0.3)
{
	specs_.recognizedGoal = base::GOAL_SAMPLEABLE_REGION;
	maxDistance_ = 0.0;
	connectionPoint_ = std::make_pair<base::State*, base::State*>(nullptr, nullptr);

	Planner::declareParam<double>("range", this, &SBL::setRange, &SBL::getRange, "0.:1.:10000.");

	defaultSettings();

	Range = custom_num;
}

ompl::geometric::SBL::~SBL()
{
	freeMemory();
}

void ompl::geometric::SBL::setup()
{

	Planner::setup();
	tools::SelfConfig sc(si_, getName());
	cout << "====================3==============\n";
	sc.configureProjectionEvaluator(projectionEvaluator_);
	cout << "======================4============\n";
	sc.configurePlannerRange(maxDistance_);
	tStart_.grid.setDimension(projectionEvaluator_->getDimension());
	tGoal_.grid.setDimension(projectionEvaluator_->getDimension());
}

void ompl::geometric::SBL::freeGridMotions(Grid<MotionInfo> &grid)
{
	for (Grid<MotionInfo>::iterator it = grid.begin(); it != grid.end() ; ++it)
	{
		for (unsigned int i = 0 ; i < it->second->data.size() ; ++i)
		{
			if (it->second->data[i]->state)
				si_->freeState(it->second->data[i]->state);
			delete it->second->data[i];
		}
	}
}

ompl::base::PlannerStatus ompl::geometric::SBL::solve(const base::PlannerTerminationCondition &ptc)
{
	Vector q(n), ik(m);
	initiate_log_parameters();
	setRange(Range); // Maximum local connection distance *** will need to profile this value

	base::State *start_node = si_->allocState();

	checkValidity();
	startTime = clock();

	base::GoalSampleableRegion *goal = dynamic_cast<base::GoalSampleableRegion*>(pdef_->getGoal().get());

	if (!goal)
	{
		OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
		return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
	}

	while (const base::State *st = pis_.nextStart())
	{
		ik = identify_state_ik(st);
		updateStateVectorIK(st, ik);
		Motion *motion = new Motion(si_);
		si_->copyState(motion->state, st);
		motion->valid = true;
		motion->root = motion->state;
		motion->ik_vect.resize(m);
		motion->ik_vect = ik;
		motion->a_chain = 0;
		addMotion(tStart_, motion);

		si_->copyState(start_node,st);
	}

	if (tStart_.size == 0)
	{
		OMPL_ERROR("%s: Motion planning start tree could not be initialized!", getName().c_str());
		return base::PlannerStatus::INVALID_START;
	}

	if (!goal->couldSample())
	{
		OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
		return base::PlannerStatus::INVALID_GOAL;
	}

	if (!sampler_)
		sampler_ = si_->allocValidStateSampler();

	OMPL_INFORM("%s: Starting planning with %d states already in datastructure", getName().c_str(), (int)(tStart_.size + tGoal_.size));

	std::vector<Motion*> solution;
	base::State *xstate = si_->allocState();

	bool      startTree = true;
	bool         solved = false;

	while (ptc == false)
	{
		TreeData &tree      = startTree ? tStart_ : tGoal_;
		startTree = !startTree;
		TreeData &otherTree = startTree ? tStart_ : tGoal_;

		// if we have not sampled too many goals already
		if (tGoal_.size == 0 || pis_.getSampledGoalsCount() < tGoal_.size / 2)
		{
			const base::State *st = tGoal_.size == 0 ? pis_.nextGoal(ptc) : pis_.nextGoal();
			if (st)
			{
				ik = identify_state_ik(st);
				updateStateVectorIK(st, ik);

				Motion *motion = new Motion(si_);
				si_->copyState(motion->state, st);
				motion->root = motion->state;
				motion->ik_vect.resize(m);
				motion->ik_vect = ik;
				motion->a_chain = 0;
				motion->valid = true;
				addMotion(tGoal_, motion);

				PlanDistance = si_->distance(start_node, motion->state);
			}
			if (tGoal_.size == 0)
			{
				OMPL_ERROR("%s: Unable to sample any valid states for goal tree", getName().c_str());
				break;
			}
		}

		Motion *existing = selectMotion(tree);
		assert(existing);
		if (!sampler_->sampleNear(xstate, existing->state, maxDistance_))
			continue;

		// Choose active chain
		active_chain = rand() % m;

		if (!IKproject(xstate, active_chain, existing->ik_vect[active_chain])) {
			project_fail++;
			continue;
		}
		project_success++;

		ik = identify_state_ik(xstate);
		updateStateVectorIK(xstate, ik);

		/* create a motion */
		Motion *motion = new Motion(si_);
		si_->copyState(motion->state, xstate);
		motion->ik_vect.resize(m);
		motion->ik_vect = ik;
		motion->a_chain = active_chain;
		motion->parent = existing;
		motion->root = existing->root;
		existing->children.push_back(motion);

		addMotion(tree, motion);

		if (checkSolution(!startTree, tree, otherTree, motion, solution))
		{
			PathGeometric *path = new PathGeometric(si_);
			for (unsigned int i = 0 ; i < solution.size() ; ++i)
				path->append(solution[i]->state);

			pdef_->addSolutionPath(base::PathPtr(path), false, 0.0, getName());
			solved = true;
			break;
		}
	}

	if (!solved)
	{
		// Report computation time
		endTime = clock();
		total_runtime = double(endTime - startTime) / CLOCKS_PER_SEC;

		nodes_in_trees = tStart_.size + tGoal_.size;
	}


	si_->freeState(xstate);

	OMPL_INFORM("%s: Created %u (%u start + %u goal) states in %u cells (%u start + %u goal)",
			getName().c_str(), tStart_.size + tGoal_.size, tStart_.size, tGoal_.size,
			tStart_.grid.size() + tGoal_.grid.size(), tStart_.grid.size(), tGoal_.grid.size());

	final_solved = solved;
	LogPerf2file(); // Log planning parameters

	return solved ? base::PlannerStatus::EXACT_SOLUTION : base::PlannerStatus::TIMEOUT;
}

bool ompl::geometric::SBL::checkSolution(bool start, TreeData &tree, TreeData &otherTree, Motion *motion, std::vector<Motion*> &solution)
{
	Grid<MotionInfo>::Coord coord;
	projectionEvaluator_->computeCoordinates(motion->state, coord);
	Grid<MotionInfo>::Cell* cell = otherTree.grid.getCell(coord);

	if (cell && !cell->data.empty())
	{
		Motion *connectOther = cell->data[rng_.uniformInt(0, cell->data.size() - 1)];

		// Check if connection is possible
		Vector ikOther = identify_state_ik(connectOther->state);
		Vector ik = motion->ik_vect;
		bool common_ik = false;
		for (int i = 0; i < ik.size(); i++)
			if (ik[i] == ikOther[i]) {
				common_ik = true;
				break;
			}
		if (!common_ik)
			return false;

		if (pdef_->getGoal()->isStartGoalPairValid(start ? motion->root : connectOther->root, start ? connectOther->root : motion->root))
		{
			Motion *connect = new Motion(si_);

			si_->copyState(connect->state, connectOther->state);
			connect->parent = motion;
			connect->root = motion->root;
			motion->children.push_back(connect);
			addMotion(tree, connect);

			if (isPathValid(tree, connect) && isPathValid(otherTree, connectOther))
			{
				// Solution found, report computation time
				endTime = clock();
				total_runtime = double(endTime - startTime) / CLOCKS_PER_SEC;
				cout << "Solved in " << total_runtime << "s." << endl;

				if (start)
					connectionPoint_ = std::make_pair(motion->state, connectOther->state);
				else
					connectionPoint_ = std::make_pair(connectOther->state, motion->state);

				/* extract the motions and put them in solution vector */

				std::vector<Motion*> mpath1;
				while (motion != nullptr)
				{
					mpath1.push_back(motion);
					motion = motion->parent;
				}

				std::vector<Motion*> mpath2;
				while (connectOther != nullptr)
				{
					mpath2.push_back(connectOther);
					connectOther = connectOther->parent;
				}

				if (!start)
					mpath1.swap(mpath2);

				save2file(mpath1, mpath2);
				cout << "Path from tree 1 size: " << mpath1.size() << ", path from tree 2 size: " << mpath2.size() << endl;
				nodes_in_path = mpath1.size() + mpath2.size();
				nodes_in_trees = tree.size + otherTree.size;

				for (int i = mpath1.size() - 1 ; i >= 0 ; --i)
					solution.push_back(mpath1[i]);
				solution.insert(solution.end(), mpath2.begin(), mpath2.end());

				return true;
			}
		}
	}
	return false;
}

bool ompl::geometric::SBL::isPathValid(TreeData &tree, Motion *motion)
{
	std::vector<Motion*> mpath;

	/* construct the solution path */
	while (motion != nullptr)
	{
		mpath.push_back(motion);
		motion = motion->parent;
	}

	/* check the path */
	for (int i = mpath.size() - 1 ; i >= 0 ; --i)
		if (!mpath[i]->valid)
		{
			bool validMotion = false;
			for (int i = 0; i < mpath[i]->ik_vect.size(); i++) {
				if (mpath[i]->parent->ik_vect[i] == mpath[i]->ik_vect[i])
					validMotion = checkMotionRBS(mpath[i]->parent->state, mpath[i]->state, i, mpath[i]->ik_vect[i]);
				if (validMotion)
					break;
			}

			if (validMotion) {
				RBS_success++;
				mpath[i]->valid = true;
			}
			else
			{
				RBS_fail++;
				removeMotion(tree, mpath[i]);
				return false;
			}
		}
	return true;
}

ompl::geometric::SBL::Motion* ompl::geometric::SBL::selectMotion(TreeData &tree)
{
	GridCell* cell = tree.pdf.sample(rng_.uniform01());
	return cell && !cell->data.empty() ? cell->data[rng_.uniformInt(0, cell->data.size() - 1)] : nullptr;
}

void ompl::geometric::SBL::removeMotion(TreeData &tree, Motion *motion)
{
	/* remove from grid */

	Grid<MotionInfo>::Coord coord;
	projectionEvaluator_->computeCoordinates(motion->state, coord);
	Grid<MotionInfo>::Cell* cell = tree.grid.getCell(coord);
	if (cell)
	{
		for (unsigned int i = 0 ; i < cell->data.size(); ++i)
		{
			if (cell->data[i] == motion)
			{
				cell->data.erase(cell->data.begin() + i);
				tree.size--;
				break;
			}
		}
		if (cell->data.empty())
		{
			tree.pdf.remove(cell->data.elem_);
			tree.grid.remove(cell);
			tree.grid.destroyCell(cell);
		}
		else
		{
			tree.pdf.update(cell->data.elem_, 1.0/cell->data.size());
		}
	}

	/* remove self from parent list */

	if (motion->parent)
	{
		for (unsigned int i = 0 ; i < motion->parent->children.size() ; ++i)
		{
			if (motion->parent->children[i] == motion)
			{
				motion->parent->children.erase(motion->parent->children.begin() + i);
				break;
			}
		}
	}

	/* remove children */
	for (unsigned int i = 0 ; i < motion->children.size() ; ++i)
	{
		motion->children[i]->parent = nullptr;
		removeMotion(tree, motion->children[i]);
	}

	if (motion->state)
		si_->freeState(motion->state);
	delete motion;
}

void ompl::geometric::SBL::addMotion(TreeData &tree, Motion *motion)
{
	Grid<MotionInfo>::Coord coord;
	projectionEvaluator_->computeCoordinates(motion->state, coord);
	Grid<MotionInfo>::Cell* cell = tree.grid.getCell(coord);
	if (cell)
	{
		cell->data.push_back(motion);
		tree.pdf.update(cell->data.elem_, 1.0/cell->data.size());
	}
	else
	{
		cell = tree.grid.createCell(coord);
		cell->data.push_back(motion);
		tree.grid.add(cell);
		cell->data.elem_ = tree.pdf.add(cell, 1.0);
	}
	tree.size++;
}

void ompl::geometric::SBL::clear()
{
	Planner::clear();

	sampler_.reset();

	freeMemory();

	tStart_.grid.clear();
	tStart_.size = 0;
	tStart_.pdf.clear();

	tGoal_.grid.clear();
	tGoal_.size = 0;
	tGoal_.pdf.clear();
	connectionPoint_ = std::make_pair<base::State*, base::State*>(nullptr, nullptr);
}

void ompl::geometric::SBL::getPlannerData(base::PlannerData &data) const
{
	Planner::getPlannerData(data);

	std::vector<MotionInfo> motions;
	tStart_.grid.getContent(motions);

	for (unsigned int i = 0 ; i < motions.size() ; ++i)
		for (unsigned int j = 0 ; j < motions[i].size() ; ++j)
			if (motions[i][j]->parent == nullptr)
				data.addStartVertex(base::PlannerDataVertex(motions[i][j]->state, 1));
			else
				data.addEdge(base::PlannerDataVertex(motions[i][j]->parent->state, 1),
						base::PlannerDataVertex(motions[i][j]->state, 1));

	motions.clear();
	tGoal_.grid.getContent(motions);
	for (unsigned int i = 0 ; i < motions.size() ; ++i)
		for (unsigned int j = 0 ; j < motions[i].size() ; ++j)
			if (motions[i][j]->parent == nullptr)
				data.addGoalVertex(base::PlannerDataVertex(motions[i][j]->state, 2));
			else
				// The edges in the goal tree are reversed so that they are in the same direction as start tree
				data.addEdge(base::PlannerDataVertex(motions[i][j]->state, 2),
						base::PlannerDataVertex(motions[i][j]->parent->state, 2));

	data.addEdge(data.vertexIndex(connectionPoint_.first), data.vertexIndex(connectionPoint_.second));
}


void ompl::geometric::SBL::save2file(vector<Motion*> mpath1, vector<Motion*> mpath2) {

	cout << "Logging path to files..." << endl;

	Vector q(n);
	Matrix path;
	int active_chain;


	// Log env. info
	std::ofstream mf;
	mf.open("./paths/path_info.txt");
	mf << n << endl << 0 << endl << getL() << endl << get_bx() << endl << get_by() << endl << get_qminmax() << endl;
	if (include_constraints)
		for (int i = 0; i < obs.size(); i++)
			for (int j = 0; j < 3; j++)
				mf << obs[i][j] << endl;
	mf.close();

	// Only milestones
	{
		// Open a_path file
		std::ofstream myfile, ikfile;
		myfile.open("./paths/path_milestones.txt");

		Vector temp;
		for (int i = mpath1.size() - 1 ; i >= 0 ; --i) {
			retrieveStateVector(mpath1[i]->state, q);
			for (int j = 0; j<n; j++) {
				myfile << q[j] << " ";
			}
			myfile << endl;

			path.push_back(q);
		}
		for (unsigned int i = 0 ; i < mpath2.size() ; ++i) {
			retrieveStateVector(mpath2[i]->state, q);
			for (int j = 0; j<n; j++) {
				myfile << q[j] << " ";
			}
			myfile << endl;

			path.push_back(q);
		}
		myfile.close();
	}

	{ // Reconstruct RBS
		// Open a_path file
		std::ofstream fp, myfile;
		std::ifstream myfile1;
		myfile.open("./paths/temp.txt",ios::out);

		std::vector<Motion*> path;

		// Bulid basic path
		for (int i = mpath1.size() - 1 ; i >= 0 ; --i)
			path.push_back(mpath1[i]);
		for (unsigned int i = 0 ; i < mpath2.size() ; ++i)
			path.push_back(mpath2[i]);

		retrieveStateVector(path[0]->state, q);
		for (int j = 0; j < q.size(); j++) {
			myfile << q[j] << " ";
		}
		myfile << endl;

		int count = 1;
		for (int i = 1; i < path.size(); i++) {

			Matrix M;
			bool valid = false;
			for (int j = 0; j < m; j++) {
				M.clear();
				if (path[i]->ik_vect[j] == path[i-1]->ik_vect[j]) {
					valid =  reconstructRBS(path[i-1]->state, path[i]->state, M, j, path[i-1]->ik_vect[j]);
				}

				if (valid)
					break;
			}

			if (!valid) {
				cout << "Error in reconstructing...\n";
				return;
			}

			for (int k = 1; k < M.size(); k++) {
				for (int j = 0; j < M[k].size(); j++) {
					myfile << M[k][j] << " ";
				}
				myfile << endl;
				count++;
			}
		}

		// Update file with number of conf.
		myfile.close();
		myfile1.open("./paths/temp.txt",ios::in);
		fp.open("./paths/path.txt",ios::out);
		fp << count << endl;
		std::string line;
		while(myfile1.good()) {
			std::getline(myfile1, line ,'\n');
			fp << line << endl;
		}
		myfile1.close();
		fp.close();
		std::remove("./paths/temp.txt");
	}
}

void ompl::geometric::SBL::LogPerf2file() {

	std::ofstream myfile;
	myfile.open("./paths/perf_log.txt");

	myfile << final_solved << " ";
	myfile << PlanDistance << " "; // Distance between nodes 1
	myfile << total_runtime << " "; // Overall planning runtime 2
	myfile << get_IK_counter() << " "; // How many IK checks? 5
	myfile << get_IK_time() << " "; // IK computation time 6
	//myfile << get_collisionCheck_counter() << endl; // How many collision checks? 7
	//myfile << get_collisionCheck_time() << endl; // Collision check computation time 8
	myfile << get_isValid_counter() << " "; // How many nodes checked 9
	myfile << nodes_in_path << " "; // Nodes in path 10
	myfile << nodes_in_trees << " "; // 11
	myfile << RBS_success << " " << RBS_fail << " ";
	myfile << project_success << " ";
	myfile << project_fail;

	myfile.close();
}
