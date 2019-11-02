package org.lenskit.mooc.svd;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixFormat;
import org.apache.commons.math3.linear.RealVector;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.bias.BiasModel;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.results.BasicResult;
import org.lenskit.results.Results;
import org.lenskit.util.collections.LongUtils;
import org.lenskit.util.keys.KeyIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * SVD-based item scorer.
 */
public class SVDItemScorer extends AbstractItemScorer {
    private static final Logger logger = LoggerFactory.getLogger(SVDItemScorer.class);
    private final SVDModel model;
    private final BiasModel baseline;
    private final DataAccessObject dao;

    /**
     * Construct an SVD item scorer using a model.
     * @param m The model to use when generating scores.
     * @param dao The data access object.
     * @param bias The baseline bias model (providing means).
     */
    @Inject
    public SVDItemScorer(SVDModel m, DataAccessObject dao,
                         BiasModel bias) {
        model = m;
        baseline = bias;
        this.dao = dao;
    }

    /**
     * Score items in a vector. The key domain of the provided vector is the
     * items to score, and the score method sets the values for each item to
     * its score (or unsets it, if no score can be provided). The previous
     * values are discarded.
     *
     * @param user   The user ID.
     * @param items The items to score
     */
    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        RealVector userFeatures = model.getUserVector(user);
        if (userFeatures == null) {
            logger.debug("unknown user {}", user);
            return Results.newResultMap();
        }

        LongSet itemSet = LongUtils.asLongSet(items);

        RealVector weights = model.getFeatureWeights();
        RealMatrix weightsMatrix = MatrixUtils.createRealDiagonalMatrix(weights.toArray());
        RealVector userVector = model.getUserVector(user);
        RealMatrix userMatrix = MatrixUtils.createRowRealMatrix(userVector.toArray());
        List<Result> results = new ArrayList<>();
        double userBaseLine = baseline.getUserBias(user);
        double intercept = baseline.getIntercept();
        // TODO Compute the predictions

        for(long item : itemSet){
            RealVector itemVector = model.getItemVector(item);
            double basline = intercept + userBaseLine + baseline.getItemBias(item);
            RealMatrix itemMatrix = MatrixUtils.createRowRealMatrix(itemVector.toArray());
            double personalized =
                    userMatrix.multiply(weightsMatrix).multiply(itemMatrix.transpose()).getEntry(0,0);
            personalized += basline;
            results.add(new BasicResult(item, personalized));
        }
        // TODO Add the predicted offsets to the baseline score
        // TODO Store the results in 'results'

        return Results.newResultMap(results);
    }
}
